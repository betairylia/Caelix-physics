using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Physics;
using Unity.Profiling;
using UnityEngine;
using Caelix.Tick;
using Caelix.Utils;

namespace Caelix.Simulation
{
    public partial class CaelixPhysicsWorld : MonoSingleton<CaelixPhysicsWorld>
    {
        private static readonly ProfilerMarker s_BrickOverlapRebuildBroadphaseMarker = new("Brick Overlap Rebuild Broadphase");
        private static readonly ProfilerMarker s_BrickOverlapQueryMarker = new("Brick Overlap Query");
        private static readonly ProfilerMarker s_BrickOverlapGraphBuildMarker = new("Brick Overlap Graph Build");
        private static readonly ProfilerMarker s_PhysicsBuildWorldMarker = new("Physics Build World");
        private static readonly ProfilerMarker s_PhysicsBeforeSimulationStartMarker = new("Physics BeforeSimulationStart");
        private static readonly ProfilerMarker s_PhysicsBuildStepInputMarker = new("Physics Build Step Input");
        private static readonly ProfilerMarker s_PhysicsDebugPreStepMarker = new("Physics Debug Pre-Step");
        private static readonly ProfilerMarker s_PhysicsBuildBroadphaseMarker = new("Physics Build Broadphase");
        private static readonly ProfilerMarker s_PhysicsCompleteBroadphaseMarker = new("Physics Complete Broadphase");
        private static readonly ProfilerMarker s_PhysicsDebugPostBroadphaseMarker = new("Physics Debug Post-Broadphase");
        private static readonly ProfilerMarker s_PhysicsResetSimulationContextMarker = new("Physics Reset Simulation Context");
        private static readonly ProfilerMarker s_PhysicsScheduleStepJobsMarker = new("Physics Schedule Step Jobs");
        private static readonly ProfilerMarker s_PhysicsCompleteStepJobsMarker = new("Physics Complete Step Jobs");
        private static readonly ProfilerMarker s_PhysicsContactDebugLoggingMarker = new("Physics Contact Debug Logging");
        private static readonly ProfilerMarker s_PhysicsOnSimulationFinishedMarker = new("Physics OnSimulationFinished");
        private static readonly ProfilerMarker s_PhysicsExportWorldMarker = new("Physics Export World");
        private static readonly ProfilerMarker s_PhysicsCompleteDisposeJobsMarker = new("Physics Complete Dispose Jobs");
        private static readonly ProfilerMarker s_PhysicsResetStaticChangedFlagMarker = new("Physics Reset Static Changed Flag");

        private readonly Dictionary<Guid128, VoxelBody> bodies = new();
        private VoxelBodyForceCommandStream bodyForceCommands;

        public IReadOnlyDictionary<Guid128, VoxelBody> Bodies => bodies;
        public VoxelBodyForceCommandStream BodyForceCommands => bodyForceCommands;

        protected PhysicsWorld physicsWorld;
        protected Unity.Physics.Simulation simulation;

        private int debugFrameCount = 0;

        [Header("Gravity")]
        public Vector3 gravity = new Vector3(0, -9.81f, 0);

        [Header("Air Friction")]
        [Tooltip("Global linear velocity damping applied to every dynamic body each step (per-second, " +
                 "exponential). Higher = lower terminal falling/sliding speed. 0 = no linear air friction.")]
        [Min(0f)] public float linearAirFriction = 0.05f;

        [Tooltip("Global angular velocity damping applied to every dynamic body each step (per-second, " +
                 "exponential). This is how rotation is damped: spinning structures reach a terminal " +
                 "angular speed just like linear drag gives a terminal linear speed. 0 = no angular air friction.")]
        [Min(0f)] public float angularAirFriction = 0.05f;

        [Header("Simulation Parameters")]
        [Tooltip("Number of substeps per simulation step")]
        public int substepCount = 1;

        [Tooltip("Number of Gauss-Seidel solver iterations")]
        public int solverIterationCount = 4;

        [Tooltip("Maximum relative velocity when separating intersecting dynamic bodies")]
        public float maxDynamicDepenetrationVelocity = 10f;

        [Tooltip("Maximum relative velocity when separating dynamic bodies from static bodies")]
        public float maxStaticDepenetrationVelocity = 10f;

        [Tooltip("Synchronize collision world after physics step (enable for precise queries within same frame)")]
        public bool synchronizeCollisionWorld = true;

        [Header("Solver Stabilization")]
        [Tooltip("Enable stabilization heuristic")]
        public bool enableSolverStabilization = true;

        [Tooltip("Velocity clipping factor")]
        public float velocityClippingFactor = 0.5f;

        [Tooltip("Inertia scaling factor")]
        public float inertiaScalingFactor = 0.75f;

        [Header("Direct Solver")]
        [Tooltip("Tuning for the accurate (Direct) solver. Only applies to contacts where BOTH bodies " +
                 "have VoxelBody.accuratePhysics enabled. If Contact Stiffness is left at 0 these are " +
                 "ignored and Unity's defaults are used instead.")]
        public Solver.DirectSolverSettings directSolverSettings = Solver.DirectSolverSettings.Default;
        public bool enableDirectSolver = true;

        [Tooltip("Multithreading enabled")] public bool multiThreaded = true;

        [Header("Brick Overlap Graph")]
        [Tooltip("Directed-record count (2x raw candidates) at or below which the post-physics " +
                 "brick-overlap graph builds in one serial job. Larger inputs are unsupported.")]
        public int brickOverlapSerialThreshold = 2048;

        [Header("Debug")]
        [Tooltip("Log per-step collision-world / broadphase diagnostics for the first few frames.")]
        public bool verboseLogging = false;

        [HideInInspector] public NativeReference<int> haveStaticBodiesChanged;

        public override void Init()
        {
            base.Init();
            physicsWorld = new PhysicsWorld(0, 0, 0);
            simulation = Unity.Physics.Simulation.Create();
            haveStaticBodiesChanged = new NativeReference<int>(0, Allocator.Persistent);
            brickOverlapGraphBuilder = new BrickOverlapGraphBuilder();
            bodyForceCommands = new VoxelBodyForceCommandStream(Allocator.Persistent);
        }

        public void AddBody(VoxelBody body)
        {
            bodies.TryAdd(body.PersistentGuid, body);
        }

        public void RemoveBody(VoxelBody body)
        {
            bodies.Remove(body.PersistentGuid);
        }

        // This step's transient body index -> stable entity GUID. Lives from the world build
        // until the next one, so post-step brick-overlap queries can resolve their candidates.
        private NativeArray<Guid128> bodyIndexToGuid;

        private BrickOverlapGraphBuilder brickOverlapGraphBuilder;

        // Time step of the last simulated step, reused when a caller asks for a broadphase
        // rebuild before querying (motion expansion must match the step that produced the pose).
        private float lastTimeStep;

        /// <summary>
        /// Read-only view of the brick-overlap graph published by the last physics step.
        /// Empty (IsCreated false) before the first step. Re-fetch each step.
        /// </summary>
        public BrickOverlapGraph BrickOverlapGraph =>
            brickOverlapGraphBuilder != null ? brickOverlapGraphBuilder.Graph : default;

        /// <summary> Counters of the last brick-overlap graph build. </summary>
        public BrickOverlapGraphStats BrickOverlapGraphStats =>
            brickOverlapGraphBuilder != null ? brickOverlapGraphBuilder.LastBuildStats : default;

        /// <summary>
        /// Queries the stepped collision world for brick overlaps, then builds and publishes
        /// the graph. Returns the published graph, or a default (empty) one when no step has
        /// run yet.
        /// </summary>
        /// <param name="queryBatches">
        /// One batch per source body, addressing the bodies of the step that just ran.
        /// </param>
        /// <param name="queryBricksByBatch">
        /// Source bricks, one stream lane per batch. Neither input is consumed nor disposed
        /// here; the caller keeps both alive until it has finished reading the graph.
        /// </param>
        /// <param name="rebuildBroadphase">
        /// Rebuilds the dynamic broadphase tree first. Not needed after a step with
        /// <see cref="synchronizeCollisionWorld"/> set — that step already synchronized the
        /// solver-integrated transforms and the BVH. Use it when the caller moved bodies since.
        /// </param>
        public BrickOverlapGraph BuildBrickOverlapGraph(
            NativeArray<VoxelBrickOverlapQueryBatch> queryBatches,
            NativeStream queryBricksByBatch,
            bool rebuildBroadphase = false,
            JobHandle inputDeps = default)
        {
            if (brickOverlapGraphBuilder == null || !bodyIndexToGuid.IsCreated)
            {
                return default;
            }

            if (rebuildBroadphase)
            {
                using (s_BrickOverlapRebuildBroadphaseMarker.Auto())
                {
                    physicsWorld.CollisionWorld.ScheduleBuildBroadphaseJobs(
                        ref physicsWorld, lastTimeStep, gravity, haveStaticBodiesChanged,
                        inputDeps, multiThreaded).Complete();
                    inputDeps = default;
                }
            }

            NativeStream rawOverlaps;
            using (s_BrickOverlapQueryMarker.Auto())
            {
                JobHandle queryHandle = physicsWorld.CollisionWorld.ScheduleVoxelBrickOverlaps(
                    queryBatches, queryBricksByBatch, out rawOverlaps, inputDeps);
                queryHandle.Complete();
            }

            using (s_BrickOverlapGraphBuildMarker.Auto())
            {
                brickOverlapGraphBuilder.serialBuildThreshold = brickOverlapSerialThreshold;
                brickOverlapGraphBuilder.BuildAndPublish(rawOverlaps, bodyIndexToGuid);
                rawOverlaps.Dispose();
            }

            return brickOverlapGraphBuilder.Graph;
        }

        public void SimulateStep(
            float dt,
            PhysicsStepInputs tickBuf)
        {
            lastTimeStep = dt;

            // The previous step's mapping stayed alive for that step's post-step queries.
            // It describes the body layout that is about to be replaced, so release it here.
            if (bodyIndexToGuid.IsCreated)
            {
                bodyIndexToGuid.Dispose();
            }

            using (s_PhysicsBuildWorldMarker.Auto())
            {
                var buildHandle = CaelixPhysicsInterface.SchedulePhysicsWorldBuild(
                    ref tickBuf, ref physicsWorld, out bodyIndexToGuid,
                    linearAirFriction, angularAirFriction, default,
                    enableDirectSolver);
                buildHandle.Complete();
                haveStaticBodiesChanged.Value = 1;
            }

            // Brick-overlap queries are not part of the step. The caller issues them against the
            // stepped, synchronized collision world through BuildBrickOverlapGraph.

            using (s_PhysicsBeforeSimulationStartMarker.Auto())
            {
                BeforeSimulationStart();
            }

            // Create solver stabilization settings
            SimulationStepInput stepInput;
            using (s_PhysicsBuildStepInputMarker.Auto())
            {
                Solver.StabilizationHeuristicSettings stabilizationSettings = enableSolverStabilization
                    ? new Solver.StabilizationHeuristicSettings
                    {
                        EnableSolverStabilization = true,
                        VelocityClippingFactor = velocityClippingFactor,
                        InertiaScalingFactor = inertiaScalingFactor
                    }
                    : Solver.StabilizationHeuristicSettings.Default;

                // A zero-initialised DirectSolverSettings means zero contact stiffness and damping, which
                // makes direct-solver contacts collapse. That is what a component serialized before this
                // field existed deserialises to, so treat it as "unset" and use Unity's defaults.
                Solver.DirectSolverSettings directSettings = directSolverSettings.ContactStiffness > 0f
                    ? directSolverSettings
                    : Solver.DirectSolverSettings.Default;

                stepInput = new SimulationStepInput()
                {
                    World = physicsWorld,
                    TimeStep = dt,
                    Gravity = gravity,
                    SynchronizeCollisionWorld = synchronizeCollisionWorld,
                    NumSubsteps = substepCount,
                    NumSolverIterations = solverIterationCount,
                    MaxDynamicDepenetrationVelocity = maxDynamicDepenetrationVelocity,
                    MaxStaticDepenetrationVelocity = maxStaticDepenetrationVelocity,
                    SolverStabilizationHeuristicSettings = stabilizationSettings,
                    DirectSolverSettings = directSettings,
                    HaveStaticBodiesChanged = haveStaticBodiesChanged
                };

                if (synchronizeCollisionWorld == false)
                {
                    Debug.LogWarning("Synchronize Collision World is disabled, brick overlap may stale");
                }
            }

            using (s_PhysicsDebugPreStepMarker.Auto())
            {
                debugFrameCount++;

                // Debug: Check collision world before simulation (first 10 frames only)
                if (verboseLogging && debugFrameCount <= 10)
                {
                    UnityEngine.Debug.Log($"[SimStep {debugFrameCount}] CollisionWorld NumBodies: {physicsWorld.CollisionWorld.NumBodies}, NumDynamic: {physicsWorld.CollisionWorld.NumDynamicBodies}, NumStatic: {physicsWorld.CollisionWorld.NumStaticBodies}");
                }
            }

            // Build the broadphase BVH trees before simulation
            using (s_PhysicsBuildBroadphaseMarker.Auto())
            {
                var buildBroadphaseHandle = physicsWorld.CollisionWorld.ScheduleBuildBroadphaseJobs(
                    ref physicsWorld, dt, gravity, haveStaticBodiesChanged, default, multiThreaded);
                using (s_PhysicsCompleteBroadphaseMarker.Auto())
                {
                    buildBroadphaseHandle.Complete();
                }
            }

            using (s_PhysicsDebugPostBroadphaseMarker.Auto())
            {
                if (verboseLogging && debugFrameCount <= 10)
                {
                    UnityEngine.Debug.Log($"[SimStep {debugFrameCount}] Broadphase built successfully");
                }
            }

            using (s_PhysicsResetSimulationContextMarker.Auto())
            {
                simulation.ResetSimulationContext(stepInput);
            }

            // Narrowphase funnel counters: clear before the jobs that fill them are scheduled.
            BeginVoxelContactProfiling();

            SimulationJobHandles handles;
            using (s_PhysicsScheduleStepJobsMarker.Auto())
            {
                handles = simulation.ScheduleStepJobs(stepInput, default, multiThreaded);
            }

            using (s_PhysicsCompleteStepJobsMarker.Auto())
            {
                handles.FinalExecutionHandle.Complete();
            }

            // Read-only contact diagnostics: must run after Complete() and before the next
            // ResetSimulationContext, while this frame's voxel contact event stream is valid.
            using (s_PhysicsContactDebugLoggingMarker.Auto())
            {
                LogVoxelContactsAfterStep(tickBuf.nDynamicBodies);
                LogVoxelContactProfileAfterStep();
            }

            using (s_PhysicsOnSimulationFinishedMarker.Auto())
            {
                OnSimulationFinished();
            }

            using (s_PhysicsExportWorldMarker.Auto())
            {
                var exportHandle = CaelixPhysicsInterface.SchedulePhysicsWorldExport(
                    ref tickBuf, ref physicsWorld, bodyIndexToGuid, default);
                exportHandle.Complete();
            }

            using (s_PhysicsCompleteDisposeJobsMarker.Auto())
            {
                handles.FinalDisposeHandle.Complete();
            }

            // Reset the static bodies changed flag after simulation
            using (s_PhysicsResetStaticChangedFlagMarker.Auto())
            {
                if (haveStaticBodiesChanged.Value > 0)
                {
                    haveStaticBodiesChanged.Value = 0;
                }
            }
        }

        public virtual void BeforeSimulationStart() { }
        public virtual void OnSimulationFinished() { }

        private void OnDisable()
        {
            simulation.Dispose();
            physicsWorld.Dispose();
            haveStaticBodiesChanged.Dispose();
            if (bodyIndexToGuid.IsCreated)
            {
                bodyIndexToGuid.Dispose();
            }
            brickOverlapGraphBuilder?.Dispose();
            brickOverlapGraphBuilder = null;
            bodyForceCommands.Dispose();

            if (verboseLogging) Debug.Log("Physics Disposed!");
        }
    }
}
