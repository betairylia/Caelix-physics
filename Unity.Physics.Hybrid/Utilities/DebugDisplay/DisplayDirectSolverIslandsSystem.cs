using Unity.Jobs;
using Unity.Burst;
using Unity.Entities;
using Unity.Transforms;
using Unity.Mathematics;
using Unity.Collections;
using Unity.DebugDisplay;
using Unity.Physics.Systems;

namespace Unity.Physics.Authoring
{
#if UNITY_EDITOR || ENABLE_UNITY_PHYSICS_RUNTIME_DEBUG_DISPLAY
    [RequireMatchingQueriesForUpdate]
    [WorldSystemFilter(WorldSystemFilterFlags.Default)]
    [UpdateInGroup(typeof(PhysicsSimulationGroup))]
    [UpdateAfter(typeof(PhysicsCreateBodyPairsGroup))]
    [UpdateBefore(typeof(PhysicsCreateContactsGroup))]
    [BurstCompile]
    partial struct DisplayDirectSolverIslandsSystem : ISystem
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
            if (physicsDebugDisplayData.DisplayDirectSolverIslands == 0)
                return;

            unsafe
            {
                m_LocalToWorldLookup.Update(ref state);

                var simulationPtr = SystemAPI.GetSingleton<SimulationSingleton>().AsSimulationPtr();
                var phasedDispatchPairs = simulationPtr->StepContext.PhasedDispatchPairs;
                var solverSchedulerInfo = simulationPtr->StepContext.SolverSchedulerInfo;
                var physicsWorld = SystemAPI.GetSingleton<PhysicsWorldSingleton>();
                var debugDraw = SystemAPI.GetSingleton<DebugDraw>();

                state.Dependency = new DisplayDirectSolverIslandsJob
                {
                    DebugDraw = debugDraw,
                    Bodies = physicsWorld.Bodies.AsReadOnly(),
                    PhasedDispatchPairs = phasedDispatchPairs.AsDeferredJobArray(),
                    SolverSchedulerInfo = solverSchedulerInfo,
                    LocalToWorldLookup = m_LocalToWorldLookup,
                    NumberSize = 0.15f,
                    NumberSpacing = 1.5f,
#if UNITY_PHYSICS_DISPLAY_ADVANCED_SOLVER_DATA
                    DisplayIslandIndex = physicsDebugDisplayData.DisplayDirectSolverIslandsIndex == 1
#endif
                }.Schedule(state.Dependency);
            }
        }
    }

    [BurstCompile]
    struct DisplayDirectSolverIslandsJob : IJob
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
        public DispatchPairSequencer.SolverSchedulerInfo SolverSchedulerInfo;
        public float NumberSize;
        public float NumberSpacing;
        public bool DisplayIslandIndex;

        [BurstCompile]
        public void Execute()
        {
            var directSolverSchedulerInfo = SolverSchedulerInfo.DirectPairsDirectScheduling;
            var firstDirectDispatchPairIndex = directSolverSchedulerInfo.FirstDispatchPairIndex.Value;
            for (int islandIndex = 0; islandIndex < directSolverSchedulerInfo.DispatchPairIslandInfoCounts.Length; ++islandIndex)
            {
                var islandColor = new ColorIndex { value = (islandIndex + 1) % ColorIndex.kMaxColors };
                var islandPairCount = directSolverSchedulerInfo.DispatchPairIslandInfoCounts[islandIndex];
                var firstIslandInfoIndex = directSolverSchedulerInfo.FirstDispatchPairIslandInfoIndices[islandIndex];
                var meanPosition = float3.zero;
                var validPairCount = 0;
                for (int i = 0; i < islandPairCount; i++)
                {
                    var islandInfoIndex = firstIslandInfoIndex + i;
                    var islandInfo = directSolverSchedulerInfo.DispatchPairIslandInfos[islandInfoIndex];
                    SafetyChecks.CheckAreEqualAndThrow(islandInfo.IslandIndex, islandIndex);

                    var unphasedDispatchPairIndex = islandInfo.UnphasedDispatchPairIndex;
                    var phasedDispatchPairIndex = firstDirectDispatchPairIndex
                        + directSolverSchedulerInfo.UnphasedToPhasedDispatchPairMap[unphasedDispatchPairIndex];

                    var phasedDispatchPair = PhasedDispatchPairs[phasedDispatchPairIndex];
                    if (!phasedDispatchPair.IsValid)
                    {
                        continue;
                    }

                    var bodyEntityA = Bodies[phasedDispatchPair.BodyIndexA].Entity;
                    var bodyEntityB = Bodies[phasedDispatchPair.BodyIndexB].Entity;
                    if (bodyEntityA != Entity.Null && bodyEntityB != Entity.Null)
                    {

                        ++validPairCount;

                        var pointA = LocalToWorldLookup[bodyEntityA].Position;
                        var pointB = LocalToWorldLookup[bodyEntityB].Position;

                        DebugDraw.Line(pointA, pointB, islandColor);

                        meanPosition += 0.5f * (pointA + pointB);
                    }
                }

                meanPosition /= validPairCount;

                if (DisplayIslandIndex)
                {
                    var numberPosition = meanPosition + (math.right() * NumberSize);
                    var spacing = NumberSize * NumberSpacing;

                    // instead of using string, this supports burst
                    FixedString32Bytes label = default;
                    label.Append('D');
                    label.Append(':');

                    DebugDraw.Text(label, numberPosition, NumberSize, spacing, islandColor);
                    DebugDraw.Number(islandIndex, numberPosition + (math.right() * 0.5f), NumberSize, spacing, islandColor);
                }
            }
        }
    }
#endif
}
