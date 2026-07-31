using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// INTERNAL UnityPhysics interface for jobs that iterate through the list of potentially
    /// overlapping body pairs produced by the broad phase Important: Only use inside UnityPhysics
    /// code! Jobs in other projects should implement IBodyPairsJob.
    /// </summary>
    [JobProducerType(typeof(IBodyPairsJobExtensions.BodyPairsJobProcess<>))]
    public interface IBodyPairsJobBase
    {
        /// <summary>   Execute operation on a given pair. </summary>
        ///
        /// <param name="pair"> [in,out] The pair. </param>
        void Execute(ref ModifiableBodyPair pair);
    }


    /// <summary>
    /// Interface for jobs that iterate through the list of potentially overlapping body pairs
    /// produced by the broad phase.
    /// </summary>
    public interface IBodyPairsJob : IBodyPairsJobBase
    {
    }

    /// <summary>   A modifiable body pair. </summary>
    public struct ModifiableBodyPair
    {
        internal EntityPair EntityPair;
        internal BodyIndexPair BodyIndexPair;

        /// <summary>   Gets the entity b. </summary>
        ///
        /// <value> The entity b. </value>
        public Entity EntityB => EntityPair.EntityB;

        /// <summary>   Gets the entity a. </summary>
        ///
        /// <value> The entity a. </value>
        public Entity EntityA => EntityPair.EntityA;

        /// <summary>   Gets the body index b. </summary>
        ///
        /// <value> The body index b. </value>
        public int BodyIndexB => BodyIndexPair.BodyIndexB;

        /// <summary>   Gets the body index a. </summary>
        ///
        /// <value> The body index a. </value>
        public int BodyIndexA => BodyIndexPair.BodyIndexA;

        /// <summary>   Disables this pair. </summary>
        public void Disable()
        {
            BodyIndexPair = BodyIndexPair.Invalid;
        }
    }

    /// <summary>   A body pairs job extensions. </summary>
    public static class IBodyPairsJobExtensions
    {
        /// <summary>   Schedules an IBodyPairsJob for serial processing. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="job">      The scheduled job. </param>
        /// <param name="simulationSingleton">  The simulation singleton. </param>
        /// <param name="world">                [in,out] The physics world. </param>
        /// <param name="inputDeps">            The input dependencies. </param>
        ///
        /// <returns>   A JobHandle. </returns>
        public static JobHandle Schedule<T>(this T job, SimulationSingleton simulationSingleton, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IBodyPairsJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }

            return ScheduleUnityPhysicsBodyPairsJob(job, simulationSingleton.AsSimulation(), ref world, inputDeps);
        }

        /// <summary>   Schedules an IBodyPairsJob for parallel processing. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="job">      The scheduled job. </param>
        /// <param name="innerLoopBatchCount">  Granularity in which workstealing is performed. A value of N, means the
        ///                                     job queue will combine N job executions and perform them in an efficient
        ///                                     inner loop.</param>
        /// <param name="simulationSingleton">  The simulation singleton. </param>
        /// <param name="world">                [in,out] The physics world. </param>
        /// <param name="inputDeps">            The input dependencies. </param>
        ///
        /// <returns>   A JobHandle. </returns>
        public static JobHandle ScheduleParallel<T>(this T job, int innerLoopBatchCount, SimulationSingleton simulationSingleton, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IBodyPairsJob
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }

            return ScheduleParallelUnityPhysicsBodyPairsJob(job, innerLoopBatchCount, simulationSingleton.AsSimulation(), ref world, inputDeps);
        }

        static unsafe JobHandle ScheduleUnityPhysicsBodyPairsJob<T>(T job, Simulation simulation, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IBodyPairsJobBase
        {
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.PostCreateBodyPairs);

            if (simulation.StepContext.PhasedDispatchPairs.IsCreated)
            {
                var data = new BodyPairsJobData<T>
                {
                    UserJobData = job,
                    PhasedDispatchPairs = simulation.StepContext.PhasedDispatchPairs.AsDeferredJobArray(),
                    Bodies = world.Bodies,
                    IsParallel = false
                };

                var jobReflectionData = BodyPairsJobProcess<T>.jobReflectionData.Data;
                BodyPairsJobProcess<T>.CheckReflectionDataCorrect(jobReflectionData);

                var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), jobReflectionData, inputDeps, ScheduleMode.Single);
                return JobsUtility.Schedule(ref parameters);
            }
            // else:

            return inputDeps;
        }

        static unsafe JobHandle ScheduleParallelUnityPhysicsBodyPairsJob<T>(this T job, int innerLoopBatchCount, Simulation simulation, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IBodyPairsJobBase
        {
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.PostCreateBodyPairs);

            if (simulation.StepContext.PhasedDispatchPairs.IsCreated)
            {
                var data = new BodyPairsJobData<T>
                {
                    UserJobData = job,
                    PhasedDispatchPairs = simulation.StepContext.PhasedDispatchPairs.AsDeferredJobArray(),
                    Bodies = world.Bodies,
                    IsParallel = true
                };

                var jobReflectionData = BodyPairsJobProcess<T>.jobReflectionData.Data;
                BodyPairsJobProcess<T>.CheckReflectionDataCorrect(jobReflectionData);

                var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), jobReflectionData, inputDeps, ScheduleMode.Parallel);
                var listDataPtr = simulation.StepContext.PhasedDispatchPairs.GetUnsafeList();
                return JobsUtility.ScheduleParallelForDeferArraySize(ref parameters, innerLoopBatchCount, listDataPtr, null);
            }

            return inputDeps;
        }

        internal struct BodyPairsJobData<T> where T : struct
        {
            public T UserJobData;
            public NativeArray<DispatchPairSequencer.DispatchPair> PhasedDispatchPairs;
            //Need to disable aliasing restriction in case T has a NativeArray of PhysicsWorld.Bodies:
            [ReadOnly][NativeDisableContainerSafetyRestriction] public NativeArray<RigidBody> Bodies;
            public bool IsParallel;
        }

        internal struct BodyPairsJobProcess<T> where T : struct, IBodyPairsJobBase
        {
            internal static readonly SharedStatic<IntPtr> jobReflectionData = SharedStatic<IntPtr>.GetOrCreate<BodyPairsJobProcess<T>>();

            [Preserve]
            internal static void Initialize()
            {
                if (jobReflectionData.Data == IntPtr.Zero)
                    jobReflectionData.Data = JobsUtility.CreateJobReflectionData(typeof(BodyPairsJobData<T>), typeof(T), (ExecuteJobFunction)Execute);
            }

            public delegate void ExecuteJobFunction(ref BodyPairsJobData<T> jobData, IntPtr additionalData,
                IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex);

            public unsafe static void Execute(ref BodyPairsJobData<T> jobData, IntPtr additionalData,
                IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex)
            {
                while (true)
                {
                    int startIndex = 0;
                    int endIndex = jobData.PhasedDispatchPairs.Length;

                    if (jobData.IsParallel)
                    {
                        if (!JobsUtility.GetWorkStealingRange(ref ranges, jobIndex, out startIndex, out endIndex))
                            break;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                        JobsUtility.PatchBufferMinMaxRanges(bufferRangePatchData, UnsafeUtility.AddressOf(ref jobData), startIndex, endIndex - startIndex);
#endif
                    }

                    for (int currentIdx = startIndex; currentIdx < endIndex; currentIdx++)
                    {
                        DispatchPairSequencer.DispatchPair dispatchPair = jobData.PhasedDispatchPairs[currentIdx];

                        // Skip joint pairs and invalid pairs
                        if (dispatchPair.IsJoint || !dispatchPair.IsValid)
                        {
                            continue;
                        }

                        var pair = new ModifiableBodyPair
                        {
                            BodyIndexPair =
                                new BodyIndexPair
                                {
                                    BodyIndexA = dispatchPair.BodyIndexA, BodyIndexB = dispatchPair.BodyIndexB
                                },
                            EntityPair = new EntityPair
                            {
                                EntityA = jobData.Bodies[dispatchPair.BodyIndexA].Entity,
                                EntityB = jobData.Bodies[dispatchPair.BodyIndexB].Entity
                            }
                        };

                        jobData.UserJobData.Execute(ref pair);

                        if (pair.BodyIndexA == -1 || pair.BodyIndexB == -1)
                        {
                            jobData.PhasedDispatchPairs[currentIdx] = DispatchPairSequencer.DispatchPair.Invalid;
                        }
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
            where T : struct, IBodyPairsJobBase
        {
            BodyPairsJobProcess<T>.Initialize();
        }
    }
}
