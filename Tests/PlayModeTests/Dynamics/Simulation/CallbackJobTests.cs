using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics.Tests.Dynamics.Simulation
{
    class CallbackJobTests : EventsTestBase
    {
        [BurstCompile]
        struct DummyBodyPairsJob : IBodyPairsJobBase
        {
            public void Execute(ref ModifiableBodyPair pair)
            {
                // does nothing
            }
        }

        [BurstCompile]
        struct DummyJacobiansJob : IJacobiansJobBase
        {
            public void Execute(ref ModifiableJacobianHeader header, ref ModifiableContactJacobian jacobian)
            {
                // does nothing
            }

            public void Execute(ref ModifiableJacobianHeader header, ref ModifiableTriggerJacobian jacobian)
            {
                // does nothing
            }
        }

        [BurstCompile]
        struct DummyContactsJob : IContactsJobBase
        {
            public void Execute(ref ModifiableContactHeader header, ref ModifiableContactPoint contact)
            {
                // does nothing
            }
        }

        [BurstCompile]
        struct BodyPairCountJob : IBodyPairsJob
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int> BodyPairCount;

            public NativeParallelHashSet<int2>.ParallelWriter BodyPairsSet;

            public int NumBodies;

            [NativeSetThreadIndex]
            int m_ThreadID;

            public void Execute(ref ModifiableBodyPair pair)
            {
                var range = new int2(0, NumBodies - 1);
                SafetyChecks.CheckInRangeAndThrow(pair.BodyIndexA, range, "BodyIndexA");
                SafetyChecks.CheckInRangeAndThrow(pair.BodyIndexB, range, "BodyIndexB");

                // make sure each body pair is unique
                var added = BodyPairsSet.TryAdd(new int2(pair.BodyIndexA, pair.BodyIndexB));
                SafetyChecks.CheckAreEqualAndThrow(true, added);

                BodyPairCount[m_ThreadID]++;
            }
        }

        [BurstCompile]
        struct ContactCountJob : IContactsJob
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int> ContactCount;

            public NativeParallelHashSet<int2>.ParallelWriter BodyPairsSet;

            public int NumBodies;

            [NativeSetThreadIndex]
            int m_ThreadID;

            public void Execute(ref ModifiableContactHeader contactHeader, ref ModifiableContactPoint contactPoint)
            {
                var range = new int2(0, NumBodies - 1);
                SafetyChecks.CheckInRangeAndThrow(contactHeader.BodyIndexA, range, "BodyIndexA");
                SafetyChecks.CheckInRangeAndThrow(contactHeader.BodyIndexB, range, "BodyIndexB");

                // make sure each body pair is unique
                if (contactPoint.Index == 0) // only check once per contact header
                {
                    var added = BodyPairsSet.TryAdd(new int2(contactHeader.BodyIndexA, contactHeader.BodyIndexB));
                    SafetyChecks.CheckAreEqualAndThrow(true, added);
                }

                ContactCount[m_ThreadID]++;
            }
        }

        [BurstCompile]
        struct JacobiansCountJob : IJacobiansJob
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int> JacobiansCount;

            public NativeParallelHashSet<int2>.ParallelWriter BodyPairsSet;

            public int NumBodies;

            [NativeSetThreadIndex]
            int m_ThreadID;

            void ValidateHeader(in ModifiableJacobianHeader header)
            {
                var range = new int2(0, NumBodies - 1);
                SafetyChecks.CheckInRangeAndThrow(header.BodyIndexA, range, "BodyIndexA");
                SafetyChecks.CheckInRangeAndThrow(header.BodyIndexB, range, "BodyIndexB");

                // make sure each body pair is unique
                var added = BodyPairsSet.TryAdd(new int2(header.BodyIndexA, header.BodyIndexB));
                SafetyChecks.CheckAreEqualAndThrow(true, added);
            }

            public void Execute(ref ModifiableJacobianHeader header, ref ModifiableContactJacobian jacobian)
            {
                ValidateHeader(header);

                JacobiansCount[m_ThreadID]++;
            }

            public void Execute(ref ModifiableJacobianHeader header, ref ModifiableTriggerJacobian jacobian)
            {
                ValidateHeader(header);

                JacobiansCount[m_ThreadID]++;
            }
        }

        static int GetTotalCount(NativeArray<int> count)
        {
            int total = 0;
            for (int i = 0; i < count.Length; ++i)
            {
                total += count[i];
            }
            return total;
        }

        /// <summary>
        /// Tests that we get the expected number of body pair, contact, and jacobian callbacks in the respective
        /// jobs regardless of the chosen solver types or threading options.
        /// </summary>
        void VerifyCollisionCallbackJobs(bool multiThreaded, SolverTypes solverTypes, bool testTriggers, bool parallelProcessing)
        {
            using (var world = new Entities.World("Test world"))
            {
                // build the physics world:

                int numBodyGroups = 50;
                var physicsWorld = CreatePhysicsWorldCausingEvents(world, numBodyGroups, solverTypes,
                    testTriggers ? EventType.Trigger : EventType.Collision, out var expectedEventCount);

                // step the simulation one frame:

                using var haveStaticBodiesChanged = new NativeReference<int>(1, Allocator.TempJob);
                var stepInput = new SimulationStepInput
                {
                    World = physicsWorld,
                    TimeStep = 1/60f,
                    Gravity = new float3(0, -9.81f, 0),
                    NumSolverIterations = 4,
                    NumSubsteps = 1,
                    DirectSolverSettings = Solver.DirectSolverSettings.Default,
                    HaveStaticBodiesChanged = haveStaticBodiesChanged
                };

                var simulation = Unity.Physics.Simulation.Create();
                var simulationSingleton = new SimulationSingleton();
                simulationSingleton.InitializeFromSimulation(ref simulation);

                using var bodyPairsCount = new NativeArray<int>(JobsUtility.JobWorkerCount + 1, Allocator.TempJob);
                using var contactCount = new NativeArray<int>(JobsUtility.JobWorkerCount + 1, Allocator.TempJob);
                using var jacobiansCount = new NativeArray<int>(JobsUtility.JobWorkerCount + 1, Allocator.TempJob);

                using var bodyPairsSet = new NativeParallelHashSet<int2>(expectedEventCount, Allocator.TempJob);
                using var contactBodyPairsSet = new NativeParallelHashSet<int2>(expectedEventCount, Allocator.TempJob);
                using var jacobiansBodyPairsCount = new NativeParallelHashSet<int2>(expectedEventCount, Allocator.TempJob);

                var bodyPairCountJob = new BodyPairCountJob
                {
                    BodyPairCount = bodyPairsCount,
                    NumBodies = physicsWorld.NumBodies,
                    BodyPairsSet = bodyPairsSet.AsParallelWriter()
                };

                var contactCountJob = new ContactCountJob
                {
                    ContactCount = contactCount,
                    NumBodies = physicsWorld.NumBodies,
                    BodyPairsSet = contactBodyPairsSet.AsParallelWriter()
                };

                var jacobiansCountJob = new JacobiansCountJob
                {
                    JacobiansCount = jacobiansCount,
                    NumBodies = physicsWorld.NumBodies,
                    BodyPairsSet = jacobiansBodyPairsCount.AsParallelWriter()
                };

                // step the system and verify that the number of body pair, contact, and jacobian callbacks received by the
                // respective jobs is as expected:

                const int kInnerLoopBatchCount = 4;

                var handles = simulation.ScheduleBroadphaseJobs(stepInput, default, multiThreaded);
                handles.FinalExecutionHandle = parallelProcessing ?
                    bodyPairCountJob.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle)
                    : bodyPairCountJob.Schedule(simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle);

                handles = simulation.ScheduleNarrowphaseJobs(stepInput, handles.FinalExecutionHandle, multiThreaded);
                handles.FinalExecutionHandle = parallelProcessing ?
                    contactCountJob.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle)
                    : contactCountJob.Schedule(simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle);

                handles = simulation.ScheduleCreateJacobiansJobs(stepInput, handles.FinalExecutionHandle, multiThreaded);
                handles.FinalExecutionHandle = parallelProcessing ?
                    jacobiansCountJob.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle)
                    : jacobiansCountJob.Schedule(simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle);

                // also complete the solve and integrate jobs to ensure proper disposal of all resources
                handles = simulation.ScheduleSolveAndIntegrateJobs(stepInput, handles.FinalExecutionHandle, multiThreaded);

                handles.FinalExecutionHandle.Complete();
                handles.FinalDisposeHandle.Complete();

                int totalBodyPairCount = GetTotalCount(bodyPairsCount);
                int totalContactCount = GetTotalCount(contactCount);
                int totalJacobiansCount = GetTotalCount(jacobiansCount);
                Assert.AreEqual(expectedEventCount, totalBodyPairCount);
                Assert.AreEqual(expectedEventCount, totalContactCount);
                Assert.AreEqual(expectedEventCount, totalJacobiansCount);

                simulation.Dispose();
            }
        }

        /// <summary>
        /// Tests that without anything to process, the scheduled jobs don't crash.
        /// </summary>
        void VerifyCollisionCallbackJobs_NothingToProcess(bool multiThreaded, bool parallelProcessing)
        {
            using (var world = new Entities.World("Test world"))
            {
                // build the physics world:
                var physicsWorld = CreatePhysicsWorldCausingNoEvents(world);
                const int kExpectedEventCount = 0;

                // step the simulation one frame:

                using var haveStaticBodiesChanged = new NativeReference<int>(1, Allocator.TempJob);
                var stepInput = new SimulationStepInput
                {
                    World = physicsWorld,
                    TimeStep = 1/60f,
                    Gravity = new float3(0, -9.81f, 0),
                    NumSolverIterations = 4,
                    NumSubsteps = 1,
                    DirectSolverSettings = Solver.DirectSolverSettings.Default,
                    HaveStaticBodiesChanged = haveStaticBodiesChanged
                };

                var simulation = Unity.Physics.Simulation.Create();
                var simulationSingleton = new SimulationSingleton();
                simulationSingleton.InitializeFromSimulation(ref simulation);

                using var bodyPairsCount = new NativeArray<int>(JobsUtility.JobWorkerCount + 1, Allocator.TempJob);
                using var contactCount = new NativeArray<int>(JobsUtility.JobWorkerCount + 1, Allocator.TempJob);
                using var jacobiansCount = new NativeArray<int>(JobsUtility.JobWorkerCount + 1, Allocator.TempJob);

                using var bodyPairsSet = new NativeParallelHashSet<int2>(kExpectedEventCount, Allocator.TempJob);
                using var contactBodyPairsSet = new NativeParallelHashSet<int2>(kExpectedEventCount, Allocator.TempJob);
                using var jacobiansBodyPairsCount = new NativeParallelHashSet<int2>(kExpectedEventCount, Allocator.TempJob);

                var bodyPairCountJob = new BodyPairCountJob
                {
                    BodyPairCount = bodyPairsCount,
                    NumBodies = physicsWorld.NumBodies,
                    BodyPairsSet = bodyPairsSet.AsParallelWriter()
                };

                var contactCountJob = new ContactCountJob
                {
                    ContactCount = contactCount,
                    NumBodies = physicsWorld.NumBodies,
                    BodyPairsSet = contactBodyPairsSet.AsParallelWriter()
                };

                var jacobiansCountJob = new JacobiansCountJob
                {
                    JacobiansCount = jacobiansCount,
                    NumBodies = physicsWorld.NumBodies,
                    BodyPairsSet = jacobiansBodyPairsCount.AsParallelWriter()
                };

                // step the system and verify that the number of body pair, contact, and jacobian callbacks received by the
                // respective jobs is as expected:

                const int kInnerLoopBatchCount = 4;

                var handles = simulation.ScheduleBroadphaseJobs(stepInput, default, multiThreaded);
                handles.FinalExecutionHandle = parallelProcessing ?
                    bodyPairCountJob.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle)
                    : bodyPairCountJob.Schedule(simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle);

                handles = simulation.ScheduleNarrowphaseJobs(stepInput, handles.FinalExecutionHandle, multiThreaded);
                handles.FinalExecutionHandle = parallelProcessing ?
                    contactCountJob.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle)
                    : contactCountJob.Schedule(simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle);

                handles = simulation.ScheduleCreateJacobiansJobs(stepInput, handles.FinalExecutionHandle, multiThreaded);
                handles.FinalExecutionHandle = parallelProcessing ?
                    jacobiansCountJob.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle)
                    : jacobiansCountJob.Schedule(simulationSingleton, ref physicsWorld, handles.FinalExecutionHandle);

                // also complete the solve and integrate jobs to ensure proper disposal of all resources
                handles = simulation.ScheduleSolveAndIntegrateJobs(stepInput, handles.FinalExecutionHandle, multiThreaded);

                handles.FinalExecutionHandle.Complete();
                handles.FinalDisposeHandle.Complete();

                int totalBodyPairCount = GetTotalCount(bodyPairsCount);
                int totalContactCount = GetTotalCount(contactCount);
                int totalJacobiansCount = GetTotalCount(jacobiansCount);
                Assert.AreEqual(kExpectedEventCount, totalBodyPairCount);
                Assert.AreEqual(kExpectedEventCount, totalContactCount);
                Assert.AreEqual(kExpectedEventCount, totalJacobiansCount);

                simulation.Dispose();
            }
        }

        [Test]
        public void ScheduleBodyPairsJob_EmptySimulation()
        {
            // Note: we have to force this stage for the scheduling below to pass the simulation stage check
            m_EmptySimulation.m_SimulationScheduleStage = SimulationScheduleStage.PostCreateBodyPairs;

            // we expect this to not crash with an empty simulation
            new DummyBodyPairsJob().Schedule(m_EmptySimulationSingleton, ref m_EmptyWorld, default)
                .Complete();
        }

        [Test]
        public void ScheduleJacobiansJob_EmptySimulation()
        {
            // Note: we have to force this stage for the scheduling below to pass the simulation stage check
            m_EmptySimulation.m_SimulationScheduleStage = SimulationScheduleStage.PostCreateJacobians;

            // we expect this to not crash with an empty simulation
            new DummyJacobiansJob().Schedule(m_EmptySimulationSingleton, ref m_EmptyWorld, default)
                .Complete();
        }

        [Test]
        public void ScheduleContactsJob_EmptySimulation()
        {
            // Note: we have to force this stage for the scheduling below to pass the simulation stage check
            m_EmptySimulation.m_SimulationScheduleStage = SimulationScheduleStage.PostCreateContacts;

            // we expect this to not crash with an empty simulation
            new DummyContactsJob().Schedule(m_EmptySimulationSingleton, ref m_EmptyWorld, default)
                .Complete();
        }

        [Test]
        public void ScheduleCollisionCallbackJobs_SerialProcessing([Values] bool multiThreaded,
            [Values] SolverTypes solverTypes, [Values] bool testTriggers) => VerifyCollisionCallbackJobs(multiThreaded, solverTypes, testTriggers, false);

        [Test]
        public void ScheduleCollisionCallbackJobs_ParallelProcessing([Values] bool multiThreaded,
            [Values] SolverTypes solverTypes, [Values] bool testTriggers) => VerifyCollisionCallbackJobs(multiThreaded, solverTypes, testTriggers, true);

        [Test]
        public void ScheduleCollisionCallbackJobs_NothingToProcess_SerialProcessing([Values] bool multiThreaded)
            => VerifyCollisionCallbackJobs_NothingToProcess(multiThreaded, false);

        [Test]
        public void ScheduleCollisionCallbackJobs_NothingToProcess_ParallelProcessing([Values] bool multiThreaded)
            => VerifyCollisionCallbackJobs_NothingToProcess(multiThreaded, true);

    }
}
