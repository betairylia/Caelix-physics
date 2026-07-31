using NUnit.Framework;
using Unity.Burst;

namespace Unity.Physics.Tests.Dynamics.Events
{
    class ImpulseEventsTests : EventsTestBase
    {
        [BurstCompile]
        struct DummyImpulseEventJob : IImpulseEventsJobBase
        {
            public void Execute(ImpulseEvent impulseEvent)
            {
                // does nothing
            }
        }

        [Test]
        public void ScheduleImpulseEventsJob_EmptySimulation()
        {
            // Note: we have to force this stage for the scheduling below to pass the simulation stage check
            m_EmptySimulation.m_SimulationScheduleStage = SimulationScheduleStage.Idle;

            // we expect this to not crash with an empty simulation
            new DummyImpulseEventJob().Schedule(m_EmptySimulationSingleton, default)
                .Complete();
        }

        /// <summary>
        /// Tests that the expected number of impulse events are exported and can be processed serially by
        /// event processing jobs regardless of the chosen solver types or threading options.
        /// </summary>
        [Test]
        public void ExportImpulseEventsTest_StepJobs_SerialProcessing([Values] bool multiThreaded, [Values] SolverTypes solverTypes)
            => VerifyExportEvents_StepJobs(multiThreaded, solverTypes, EventType.Impulse, parallelEventProcessing: false);

        /// <summary>
        /// Tests that the expected number of impulse events are exported and can be processed in parallel by
        /// event processing jobs regardless of the chosen solver types or threading options.
        /// </summary>
        [Test]
        public void ExportImpulseEventsTest_StepJobs_ParallelProcessing([Values] bool multiThreaded, [Values] SolverTypes solverTypes)
            => VerifyExportEvents_StepJobs(multiThreaded, solverTypes, EventType.Impulse, parallelEventProcessing: true);

        /// <summary>
        /// Tests that the expected number of impulse events are exported in immediate mode regardless of the chosen
        /// solver types.
        /// </summary>
        [Test]
        public void ExportImpulseEventsTest_StepImmediate([Values] SolverTypes solverTypes)
            => VerifyExportEvents_StepImmediate(solverTypes, EventType.Impulse);
    }
}
