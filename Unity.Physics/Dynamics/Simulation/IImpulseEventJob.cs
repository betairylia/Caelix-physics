using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>   Interface for impulse events job base. </summary>
    [JobProducerType(typeof(IImpulseEventJobExtensions.ImpulseEventJobProcess<>))]
    public interface IImpulseEventsJobBase
    {
        /// <summary>   Executes an operation on the given impulse event. </summary>
        ///
        /// <param name="impulseEvent"> The impulse event. </param>
        void Execute(ImpulseEvent impulseEvent);
    }

    /// <summary>   Interface for impulse events job. </summary>
    public interface IImpulseEventsJob : IImpulseEventsJobBase
    {
    }

    /// <summary>   An extension class for scheduling ImpulseEventsJob. </summary>
    public static class IImpulseEventJobExtensions
    {
        /// <summary>   Schedules an IImpulseEventsJob for serial processing. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="job">      The scheduled job. </param>
        /// <param name="simulationSingleton">  The simulation singleton. </param>
        /// <param name="inputDeps">            The input dependencies. </param>
        ///
        /// <returns>   A JobHandle. </returns>
        public static JobHandle Schedule<T>(this T job, SimulationSingleton simulationSingleton, JobHandle inputDeps)
            where T : struct, IImpulseEventsJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }
            return ScheduleUnityPhysicsImpulseEventsJob(job, simulationSingleton.AsSimulation(), inputDeps);
        }

        /// <summary>   Schedules an IImpulseEventsJob for parallel processing. </summary>
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
            where T : struct, IImpulseEventsJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }
            return ScheduleParallelUnityPhysicsImpulseEventsJob(job, innerLoopBatchCount, simulationSingleton.AsSimulation(), inputDeps);
        }

        static unsafe JobHandle ScheduleUnityPhysicsImpulseEventsJob<T>(T job, Simulation simulation, JobHandle inputDeps)
            where T : struct, IImpulseEventsJobBase
        {
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.Idle);
            if (!simulation.ReadyForEventScheduling)
                return inputDeps;

            var data = new ImpulseEventJobData<T>
            {
                UserJobData = job,
                EventReader = simulation.ImpulseEvents.EventDataStream.AsReader(),
                IsParallel = false
            };

            var jobReflectionData = ImpulseEventJobProcess<T>.jobReflectionData.Data;
            ImpulseEventJobProcess<T>.CheckReflectionDataCorrect(jobReflectionData);

            var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), jobReflectionData, inputDeps, ScheduleMode.Single);
            return JobsUtility.Schedule(ref parameters);
        }

        static unsafe JobHandle ScheduleParallelUnityPhysicsImpulseEventsJob<T>(T jobData, int innerLoopBatchCount, Simulation simulation, JobHandle inputDeps)
            where T : struct, IImpulseEventsJobBase
        {
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.Idle);
            if (!simulation.ReadyForEventScheduling)
                return inputDeps;

            var eventDataStream = simulation.ImpulseEvents.EventDataStream;
            var data = new ImpulseEventJobData<T>
            {
                UserJobData = jobData,
                EventReader = eventDataStream.AsReader(),
                IsParallel = true
            };

            var jobReflectionData = ImpulseEventJobProcess<T>.jobReflectionData.Data;
            ImpulseEventJobProcess<T>.CheckReflectionDataCorrect(jobReflectionData);

            var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), jobReflectionData, inputDeps, ScheduleMode.Parallel);
            var forEachCountPtr = NativeStreamUnsafeUtility.GetUnsafeForEachCountPtr(ref eventDataStream);
            var listDataPtr = (byte*)forEachCountPtr - sizeof(void*);
            return JobsUtility.ScheduleParallelForDeferArraySize(ref parameters, innerLoopBatchCount, listDataPtr, null);
        }

        internal struct ImpulseEventJobData<T> where T : struct
        {
            public T UserJobData;
            [NativeDisableContainerSafetyRestriction] public NativeStream.Reader EventReader;
            public bool IsParallel;
        }

        internal struct ImpulseEventJobProcess<T> where T : struct, IImpulseEventsJobBase
        {
            internal static readonly SharedStatic<IntPtr> jobReflectionData = SharedStatic<IntPtr>.GetOrCreate<ImpulseEventJobProcess<T>>();

            [Preserve]
            public static void Initialize()
            {
                if (jobReflectionData.Data == IntPtr.Zero)
                    jobReflectionData.Data = JobsUtility.CreateJobReflectionData(typeof(ImpulseEventJobData<T>), typeof(T), (ExecuteJobFunction)Execute);
            }

            [System.Diagnostics.Conditional("ENABLE_UNITY_COLLECTIONS_CHECK")]
            internal static void CheckReflectionDataCorrect(IntPtr reflectionData)
            {
                if (reflectionData == IntPtr.Zero)
                    throw new InvalidOperationException("Reflection data was not set up by an Initialize() call");
            }

            public delegate void ExecuteJobFunction(ref ImpulseEventJobData<T> jobData, IntPtr additionalData,
                IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex);

            public unsafe static void Execute(ref ImpulseEventJobData<T> jobData, IntPtr additionalData,
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

                    var eventEnumerator = new ImpulseEvents.Enumerator(jobData.EventReader, forEachIndexBegin, forEachIndexEnd);

                    while (eventEnumerator.MoveNext())
                    {
                        jobData.UserJobData.Execute(eventEnumerator.Current);
                    }

                    // If we are not running in parallel, we are done.
                    if (!jobData.IsParallel)
                        break;
                }
            }
        }

        /// <summary>   Early job initialize. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        public static void EarlyJobInit<T>()
            where T : struct, IImpulseEventsJobBase
        {
            ImpulseEventJobProcess<T>.Initialize();
        }
    }
}
