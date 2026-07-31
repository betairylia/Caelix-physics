using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// INTERNAL UnityPhysics interface for jobs that iterate through the list of collision events
    /// produced by the solver. Important: Only use inside UnityPhysics code! Jobs in other projects
    /// should implement ICollisionEventsJob.
    /// </summary>
    [JobProducerType(typeof(ICollisionEventJobExtensions.CollisionEventJobProcess<>))]
    public interface ICollisionEventsJobBase
    {
        /// <summary>   Executes the operation on a given collision event. </summary>
        ///
        /// <param name="collisionEvent">   The collision event. </param>
        void Execute(CollisionEvent collisionEvent);
    }


    /// <summary>
    /// Interface for jobs that iterate through the list of collision events produced by the solver.
    /// </summary>
    public interface ICollisionEventsJob : ICollisionEventsJobBase
    {
    }


    /// <summary>   A collision event job extensions. </summary>
    public static class ICollisionEventJobExtensions
    {
        /// <summary>   Schedules an ICollisionEventsJob for serial processing. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="job">      The scheduled job. </param>
        /// <param name="simulationSingleton">  The simulation singleton. </param>
        /// <param name="inputDeps">            The input dependencies. </param>
        ///
        /// <returns>   A JobHandle. </returns>
        public static JobHandle Schedule<T>(this T job, SimulationSingleton simulationSingleton, JobHandle inputDeps)
            where T : struct, ICollisionEventsJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }

            return ScheduleUnityPhysicsCollisionEventsJob(job, simulationSingleton.AsSimulation(), inputDeps);
        }

        /// <summary>   Schedules an ICollisionEventsJob for parallel processing. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="job">      The scheduled job. </param>
        /// <param name="innerLoopBatchCount">  Granularity in which workstealing is performed. A value of N, means the
        ///                                     job queue will combine N job executions and perform them in an efficient
        ///                                     inner loop.</param>
        /// <param name="simulationSingleton">  The simulation singleton. </param>
        /// <param name="inputDeps">            The input dependencies. </param>
        ///
        /// <returns>   A JobHandle. </returns>
        public static JobHandle ScheduleParallel<T>(this T job, int innerLoopBatchCount, SimulationSingleton simulationSingleton, JobHandle inputDeps)
            where T : struct, ICollisionEventsJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }

            return ScheduleParallelUnityPhysicsCollisionEventsJob(job, innerLoopBatchCount, simulationSingleton.AsSimulation(), inputDeps);
        }

        static unsafe JobHandle ScheduleUnityPhysicsCollisionEventsJob<T>(T job, Simulation simulation, JobHandle inputDeps)
            where T : struct, ICollisionEventsJobBase
        {
            // Idle means before or after simulation, which is fine in 99% of cases - the one case where we have trouble is the following:
            // Sim type == Unity.Physics
            // The simulation hasn't run at least once (can happen if we put [UpdateBefore(typeof(PhysicsCreateBodyPairsGroup)] on the first frame, so we need extra checks
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.Idle);
            if (!simulation.ReadyForEventScheduling)
                return inputDeps;

            var data = new CollisionEventJobData<T>
            {
                UserJobData = job,
                EventReader = simulation.CollisionEvents.EventDataStream.AsReader(),
                InputVelocities = simulation.CollisionEvents.InputVelocities,
                TimeStep = simulation.CollisionEvents.TimeStep,
                IsParallel = false
            };

            var jobReflectionData = CollisionEventJobProcess<T>.jobReflectionData.Data;
            CollisionEventJobProcess<T>.CheckReflectionDataCorrect(jobReflectionData);

            var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), jobReflectionData, inputDeps, ScheduleMode.Single);
            return JobsUtility.Schedule(ref parameters);
        }

        static unsafe JobHandle ScheduleParallelUnityPhysicsCollisionEventsJob<T>(T job, int innerLoopBatchCount, Simulation simulation, JobHandle inputDeps)
            where T : struct, ICollisionEventsJobBase
        {
            // Idle means before or after simulation, which is fine in 99% of cases - the one case where we have trouble is the following:
            // Sim type == Unity.Physics
            // The simulation hasn't run at least once (can happen if we put [UpdateBefore(typeof(PhysicsCreateBodyPairsGroup)] on the first frame, so we need extra checks
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.Idle);
            if (!simulation.ReadyForEventScheduling)
                return inputDeps;

            var eventDataStream = simulation.CollisionEvents.EventDataStream;
            var data = new CollisionEventJobData<T>
            {
                UserJobData = job,
                EventReader = eventDataStream.AsReader(),
                InputVelocities = simulation.CollisionEvents.InputVelocities,
                TimeStep = simulation.CollisionEvents.TimeStep,
                IsParallel = true
            };

            var jobReflectionData = CollisionEventJobProcess<T>.jobReflectionData.Data;
            CollisionEventJobProcess<T>.CheckReflectionDataCorrect(jobReflectionData);

            var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), jobReflectionData, inputDeps, ScheduleMode.Parallel);
            var forEachCountPtr = NativeStreamUnsafeUtility.GetUnsafeForEachCountPtr(ref eventDataStream);
            var listDataPtr = (byte*)forEachCountPtr - sizeof(void*);
            return JobsUtility.ScheduleParallelForDeferArraySize(ref parameters, innerLoopBatchCount, listDataPtr, null);
        }

        internal struct CollisionEventJobData<T> where T : struct
        {
            public T UserJobData;

            [NativeDisableContainerSafetyRestriction]
            public NativeStream.Reader EventReader;
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<Velocity> InputVelocities;
            public float TimeStep;
            public bool IsParallel;
        }

        internal struct CollisionEventJobProcess<T> where T : struct, ICollisionEventsJobBase
        {
            internal static readonly SharedStatic<IntPtr> jobReflectionData = SharedStatic<IntPtr>.GetOrCreate<CollisionEventJobProcess<T>>();

            [Preserve]
            public static void Initialize()
            {
                if (jobReflectionData.Data == IntPtr.Zero)
                    jobReflectionData.Data = JobsUtility.CreateJobReflectionData(typeof(CollisionEventJobData<T>), typeof(T), (ExecuteJobFunction)Execute);
            }

            public delegate void ExecuteJobFunction(ref CollisionEventJobData<T> jobData, IntPtr additionalData, IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex);

            public unsafe static void Execute(ref CollisionEventJobData<T> jobData, IntPtr additionalData,
                IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex)
            {
                while (true)
                {
                    int forEachIndexBegin = 0;
                    int forEachIndexEnd = jobData.EventReader.ForEachCount;

                    if (jobData.IsParallel)
                    {
                        if (!JobsUtility.GetWorkStealingRange(ref ranges, jobIndex, out forEachIndexBegin,
                                out forEachIndexEnd))
                            break;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                        JobsUtility.PatchBufferMinMaxRanges(bufferRangePatchData, UnsafeUtility.AddressOf(ref jobData),
                            forEachIndexBegin, forEachIndexEnd - forEachIndexBegin);
#endif
                    }

                    var eventEnumerator = new CollisionEvents.Enumerator(jobData.EventReader,
                        jobData.InputVelocities, jobData.TimeStep, forEachIndexBegin, forEachIndexEnd);

                    while (eventEnumerator.MoveNext())
                    {
                        jobData.UserJobData.Execute(eventEnumerator.Current);
                    }

                    // If we are not running in parallel, we are done.
                    if (!jobData.IsParallel)
                        break;
                }
            }

            [System.Diagnostics.Conditional("ENABLE_UNITY_COLLECTIONS_CHECK")]
            internal static void CheckReflectionDataCorrect(IntPtr reflectionData)
            {
                if (reflectionData == IntPtr.Zero)
                    SafetyChecks.ThrowInvalidOperationException("Reflection data was not set up by an Initialize() call");
            }
        }

        /// <summary>   Early job initialize. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        public static void EarlyJobInit<T>()
            where T : struct, ICollisionEventsJobBase
        {
            CollisionEventJobProcess<T>.Initialize();
        }
    }
}
