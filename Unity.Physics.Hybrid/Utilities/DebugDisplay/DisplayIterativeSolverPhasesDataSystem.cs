using Unity.Burst;
using Unity.Collections;
using Unity.DebugDisplay;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Systems;
using Unity.Transforms;
using static Unity.Physics.DispatchPairSequencer.IterativeSolverSchedulerInfo;

namespace Unity.Physics.Authoring
{
#if UNITY_EDITOR || ENABLE_UNITY_PHYSICS_RUNTIME_DEBUG_DISPLAY
    [RequireMatchingQueriesForUpdate]
    [WorldSystemFilter(WorldSystemFilterFlags.Default)]
    [UpdateInGroup(typeof(PhysicsSimulationGroup))]
    [UpdateAfter(typeof(PhysicsCreateBodyPairsGroup))]
    [UpdateBefore(typeof(PhysicsCreateContactsGroup))]
    [BurstCompile]
    partial struct DisplayIterativeSolverPhasesSystem : ISystem
    {
        ComponentLookup<LocalToWorld> m_LocalToWorldLookup;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<PhysicsDebugDisplayData>();
            state.RequireForUpdate<PhysicsWorldSingleton>();
            state.RequireForUpdate<SimulationSingleton>();
            state.RequireForUpdate<DebugDraw>();

            m_LocalToWorldLookup = SystemAPI.GetComponentLookup<LocalToWorld>(true);
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var physicsDebugDisplayData = SystemAPI.GetSingleton<PhysicsDebugDisplayData>();
            if (physicsDebugDisplayData.DrawIterativeSolverPhases == 0)
                return;

            unsafe
            {
                m_LocalToWorldLookup.Update(ref state);

                var simulationPtr = SystemAPI.GetSingleton<SimulationSingleton>().AsSimulationPtr();
                var physicsWorld = SystemAPI.GetSingleton<PhysicsWorldSingleton>();
                var debugDraw = SystemAPI.GetSingleton<DebugDraw>();

                var colorsPerType = ColorIndex.kMaxColors / 3;

                var displayIterativePairsHandle = new DisplayIterativeSolverPhasesJob
                {
                    NumberSize = 0.15f,
                    NumberSpacing = 1.5f,
                    ColorIndexOffset = 0,
                    DebugDraw = debugDraw,
                    Bodies = physicsWorld.Bodies.AsReadOnly(),
                    LocalToWorldLookup = m_LocalToWorldLookup,
                    PhasedDispatchPairs = simulationPtr->StepContext.PhasedDispatchPairs.AsDeferredJobArray(),
                    IterativeSolverSchedulerInfo = simulationPtr->StepContext.SolverSchedulerInfo.IterativePairsIterativeScheduling
#if UNITY_PHYSICS_DISPLAY_ADVANCED_SOLVER_DATA
                    DrawIterativeSolverIndex = physicsDebugDisplayData.DrawIterativeSolverIndex != 0,
#endif
                }.Schedule(state.Dependency);

                var displayCouplingPairsHandle = new DisplayIterativeSolverPhasesJob
                {
                    NumberSize = 0.15f,
                    NumberSpacing = 1.5f,
                    ColorIndexOffset = colorsPerType,
                    DebugDraw = debugDraw,
                    Bodies = physicsWorld.Bodies.AsReadOnly(),
                    LocalToWorldLookup = m_LocalToWorldLookup,
                    PhasedDispatchPairs = simulationPtr->StepContext.PhasedDispatchPairs.AsDeferredJobArray(),
                    IterativeSolverSchedulerInfo = simulationPtr->StepContext.SolverSchedulerInfo.CouplingPairsIterativeScheduling
#if UNITY_PHYSICS_DISPLAY_ADVANCED_SOLVER_DATA
                    DrawIterativeSolverIndex = physicsDebugDisplayData.DrawIterativeSolverIndex != 0,
#endif
                }.Schedule(state.Dependency);

                var displayDirectPairsHandle = new DisplayIterativeSolverPhasesJob
                {
                    NumberSize = 0.15f,
                    NumberSpacing = 1.5f,
                    ColorIndexOffset = colorsPerType * 2,
                    DebugDraw = debugDraw,
                    Bodies = physicsWorld.Bodies.AsReadOnly(),
                    LocalToWorldLookup = m_LocalToWorldLookup,
                    PhasedDispatchPairs = simulationPtr->StepContext.PhasedDispatchPairs.AsDeferredJobArray(),
                    IterativeSolverSchedulerInfo = simulationPtr->StepContext.SolverSchedulerInfo.DirectPairsIterativeScheduling
#if UNITY_PHYSICS_DISPLAY_ADVANCED_SOLVER_DATA
                    DrawIterativeSolverIndex = physicsDebugDisplayData.DrawIterativeSolverIndex != 0,
#endif
                }.Schedule(state.Dependency);

                state.Dependency = JobHandle.CombineDependencies(displayIterativePairsHandle, displayCouplingPairsHandle, displayDirectPairsHandle);
            }
        }
    }

    [BurstCompile]
    struct DisplayIterativeSolverPhasesJob : IJob
    {
        [ReadOnly]
        public DebugDraw DebugDraw;
        [ReadOnly]
        public ComponentLookup<LocalToWorld> LocalToWorldLookup;
        [ReadOnly]
        public NativeArray<RigidBody>.ReadOnly Bodies;
        [ReadOnly]
        public NativeArray<DispatchPairSequencer.DispatchPair> PhasedDispatchPairs;
        [ReadOnly]
        public DispatchPairSequencer.IterativeSolverSchedulerInfo IterativeSolverSchedulerInfo;

        public float NumberSize;
        public float NumberSpacing;
        public int ColorIndexOffset;
        public bool DrawIterativeSolverIndex;

        [BurstCompile]
        public void Execute()
        {
            var firstDispatchPairIndex = IterativeSolverSchedulerInfo.FirstDispatchPairIndex.Value;
            for (int phaseIndex = 0; phaseIndex < IterativeSolverSchedulerInfo.PhaseInfo.Length; phaseIndex++)
            {
                var phaseColor = new ColorIndex { value = (ColorIndexOffset + phaseIndex + 1) % ColorIndex.kMaxColors };
                var solvePhaseInfo = IterativeSolverSchedulerInfo.PhaseInfo[phaseIndex];
                var firstIndex = firstDispatchPairIndex + solvePhaseInfo.FirstDispatchPairOffset;
                var lastIndex = firstIndex + solvePhaseInfo.DispatchPairCount;
                var avgPosition = float3.zero;
                var numPoints = 0;

                for (int j = firstIndex; j < lastIndex; j++)
                {
                    var item = PhasedDispatchPairs[j];
                    if (!item.IsValid)
                    {
                        continue;
                    }

                    var bodyA = Bodies[item.BodyIndexA].Entity;
                    var bodyB = Bodies[item.BodyIndexB].Entity;

                    if (bodyA != Entity.Null && bodyB != Entity.Null)
                    {
                        var pointA = LocalToWorldLookup[bodyA].Position;
                        var pointB = LocalToWorldLookup[bodyB].Position;

                        DebugDraw.Line(pointA, pointB, phaseColor);

                        avgPosition += pointA;
                        ++numPoints;
                    }
                }

                if (DrawIterativeSolverIndex)
                {
                    var position = avgPosition / numPoints;
                    var numberPosition = position + (math.right() * NumberSize);
                    var spacing = NumberSize * NumberSpacing;

                    // instead of using string, this supports burst
                    FixedString32Bytes label = default;
                    label.Append('I');
                    label.Append(':');

                    DebugDraw.Text(label, numberPosition, NumberSize, spacing, phaseColor);
                    DebugDraw.Number(phaseIndex + 1, numberPosition + (math.right() * 0.5f), NumberSize, spacing, phaseColor);
                }
            }
        }
    }
#endif
}
