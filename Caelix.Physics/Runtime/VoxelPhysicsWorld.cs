using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Profiling;
using UnityEngine;
using Caelix.Utils;

namespace Caelix.Simulation
{
    /// <summary>
    /// Every tunable of one <see cref="VoxelPhysicsWorld"/>. Plain data so a world config can carry
    /// it and a bootstrap component can push inspector edits into a running world each frame.
    /// </summary>
    [Serializable]
    public struct PhysicsWorldSettings
    {
        public float3 gravity;

        /// <summary>Global linear velocity damping per second, exponential. 0 = none.</summary>
        public float linearAirFriction;

        /// <summary>Global angular velocity damping per second, exponential. 0 = none.</summary>
        public float angularAirFriction;

        public int substepCount;
        public int solverIterationCount;
        public float maxDynamicDepenetrationVelocity;
        public float maxStaticDepenetrationVelocity;
        public bool synchronizeCollisionWorld;

        public bool enableSolverStabilization;
        public float velocityClippingFactor;
        public float inertiaScalingFactor;

        /// <summary>
        /// Tuning for the accurate (Direct) solver. Only applies to contacts where both bodies
        /// opted in. A zero contact stiffness means "unset": Unity's defaults are used instead.
        /// </summary>
        public Solver.DirectSolverSettings directSolverSettings;
        public bool enableDirectSolver;
        public bool multiThreaded;

        /// <summary>
        /// Directed-record count (2x raw candidates) at or below which the post-physics
        /// brick-overlap graph builds in one serial job.
        /// </summary>
        public int brickOverlapSerialThreshold;

        public bool verboseLogging;

        // Contact debug logging (see VoxelPhysicsWorld.ContactDebug.cs)
        public bool enableContactDebugLogging;
        public float contactDebugSpikeSpeedDelta;

        // Contact profiling (see VoxelPhysicsWorld.ContactProfiling.cs)
        public bool enableContactProfiling;
        public int contactProfilingLogInterval;
        public bool contactProfilingAverage;

        public static PhysicsWorldSettings Default => new PhysicsWorldSettings
        {
            gravity = new float3(0f, -9.81f, 0f),
            linearAirFriction = 0.05f,
            angularAirFriction = 0.05f,
            substepCount = 1,
            solverIterationCount = 4,
            maxDynamicDepenetrationVelocity = 10f,
            maxStaticDepenetrationVelocity = 10f,
            synchronizeCollisionWorld = true,
            enableSolverStabilization = true,
            velocityClippingFactor = 0.5f,
            inertiaScalingFactor = 0.75f,
            directSolverSettings = Solver.DirectSolverSettings.Default,
            enableDirectSolver = true,
            multiThreaded = true,
            brickOverlapSerialThreshold = 2048,
            verboseLogging = false,
            enableContactDebugLogging = false,
            contactDebugSpikeSpeedDelta = 1.0f,
            enableContactProfiling = false,
            contactProfilingLogInterval = 60,
            contactProfilingAverage = true,
        };
    }

    /// <summary>
    /// The voxel physics simulation of one world: a Unity Physics <see cref="PhysicsWorld"/> plus
    /// its <see cref="Unity.Physics.Simulation"/>, the brick-overlap graph builder, and the force
    /// command stream. A plain class owned by the world; no GameObject and no singleton.
    /// </summary>
    public sealed partial class VoxelPhysicsWorld : IDisposable
    {
        private static readonly ProfilerMarker s_BrickOverlapRebuildBroadphaseMarker = new("Brick Overlap Rebuild Broadphase");
        private static readonly ProfilerMarker s_BrickOverlapQueryMarker = new("Brick Overlap Query");
        private static readonly ProfilerMarker s_BrickOverlapGraphBuildMarker = new("Brick Overlap Graph Build");
        private static readonly ProfilerMarker s_PhysicsBuildWorldMarker = new("Physics Build World");
        private static readonly ProfilerMarker s_PhysicsBuildStepInputMarker = new("Physics Build Step Input");
        private static readonly ProfilerMarker s_PhysicsBuildBroadphaseMarker = new("Physics Build Broadphase");
        private static readonly ProfilerMarker s_PhysicsCompleteBroadphaseMarker = new("Physics Complete Broadphase");
        private static readonly ProfilerMarker s_PhysicsResetSimulationContextMarker = new("Physics Reset Simulation Context");
        private static readonly ProfilerMarker s_PhysicsScheduleStepJobsMarker = new("Physics Schedule Step Jobs");
        private static readonly ProfilerMarker s_PhysicsCompleteStepJobsMarker = new("Physics Complete Step Jobs");
        private static readonly ProfilerMarker s_PhysicsContactDebugLoggingMarker = new("Physics Contact Debug Logging");
        private static readonly ProfilerMarker s_PhysicsExportWorldMarker = new("Physics Export World");
        private static readonly ProfilerMarker s_PhysicsCompleteDisposeJobsMarker = new("Physics Complete Dispose Jobs");

        /// <summary>Live-tunable settings. Read at the start of every step.</summary>
        public PhysicsWorldSettings Settings;

        private PhysicsWorld physicsWorld;
        private Unity.Physics.Simulation simulation;
        private NativeReference<int> haveStaticBodiesChanged;
        private readonly VoxelBodyForceCommandStream bodyForceCommands;
        private BrickOverlapGraphBuilder brickOverlapGraphBuilder;
        private int debugFrameCount;
        private bool disposed;

        // This step's transient body index -> stable entity GUID. Lives from the world build
        // until the next one, so post-step brick-overlap queries can resolve their candidates.
        private NativeArray<Guid128> bodyIndexToGuid;

        // Time step of the last simulated step, reused when a caller asks for a broadphase
        // rebuild before querying (motion expansion must match the step that produced the pose).
        private float lastTimeStep;

        public VoxelBodyForceCommandStream BodyForceCommands => bodyForceCommands;

        /// <summary>The Unity Physics world of the last step. Read-only diagnostics only.</summary>
        public ref readonly PhysicsWorld PhysicsWorld => ref physicsWorld;

        /// <summary>
        /// Read-only view of the brick-overlap graph published by the last build. Empty
        /// (IsCreated false) before the first build. Re-fetch each step.
        /// </summary>
        public BrickOverlapGraph BrickOverlapGraph =>
            brickOverlapGraphBuilder != null ? brickOverlapGraphBuilder.Graph : default;

        /// <summary>Counters of the last brick-overlap graph build.</summary>
        public BrickOverlapGraphStats BrickOverlapGraphStats =>
            brickOverlapGraphBuilder != null ? brickOverlapGraphBuilder.LastBuildStats : default;

        public VoxelPhysicsWorld(PhysicsWorldSettings settings)
        {
            Settings = settings;
            physicsWorld = new PhysicsWorld(0, 0, 0);
            simulation = Unity.Physics.Simulation.Create();
            haveStaticBodiesChanged = new NativeReference<int>(0, Allocator.Persistent);
            brickOverlapGraphBuilder = new BrickOverlapGraphBuilder();
            bodyForceCommands = new VoxelBodyForceCommandStream(Allocator.Persistent);
        }

        /// <summary>
        /// Queries the stepped collision world for brick overlaps, then builds and publishes
        /// the graph. Returns the published graph, or a default (empty) one when no step has
        /// run yet.
        /// </summary>
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
                        ref physicsWorld, lastTimeStep, Settings.gravity, haveStaticBodiesChanged,
                        inputDeps, Settings.multiThreaded).Complete();
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
                brickOverlapGraphBuilder.serialBuildThreshold = Settings.brickOverlapSerialThreshold;
                brickOverlapGraphBuilder.BuildAndPublish(rawOverlaps, bodyIndexToGuid);
                rawOverlaps.Dispose();
            }

            return brickOverlapGraphBuilder.Graph;
        }

        /// <summary>
        /// Runs one physics step over the world's entities and bodies. Reads transforms and
        /// velocities from <paramref name="tickBuf"/> and writes the stepped poses and velocities
        /// back into it.
        /// </summary>
        public void SimulateStep(float dt, ref PhysicsStepInputs tickBuf)
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
                    Settings.linearAirFriction, Settings.angularAirFriction, default,
                    Settings.enableDirectSolver);
                buildHandle.Complete();
                haveStaticBodiesChanged.Value = 1;
            }

            SimulationStepInput stepInput;
            using (s_PhysicsBuildStepInputMarker.Auto())
            {
                Solver.StabilizationHeuristicSettings stabilizationSettings = Settings.enableSolverStabilization
                    ? new Solver.StabilizationHeuristicSettings
                    {
                        EnableSolverStabilization = true,
                        VelocityClippingFactor = Settings.velocityClippingFactor,
                        InertiaScalingFactor = Settings.inertiaScalingFactor
                    }
                    : Solver.StabilizationHeuristicSettings.Default;

                // A zero-initialised DirectSolverSettings means zero contact stiffness and damping, which
                // makes direct-solver contacts collapse. That is what a component serialized before this
                // field existed deserialises to, so treat it as "unset" and use Unity's defaults.
                Solver.DirectSolverSettings directSettings = Settings.directSolverSettings.ContactStiffness > 0f
                    ? Settings.directSolverSettings
                    : Solver.DirectSolverSettings.Default;

                stepInput = new SimulationStepInput()
                {
                    World = physicsWorld,
                    TimeStep = dt,
                    Gravity = Settings.gravity,
                    SynchronizeCollisionWorld = Settings.synchronizeCollisionWorld,
                    NumSubsteps = Settings.substepCount,
                    NumSolverIterations = Settings.solverIterationCount,
                    MaxDynamicDepenetrationVelocity = Settings.maxDynamicDepenetrationVelocity,
                    MaxStaticDepenetrationVelocity = Settings.maxStaticDepenetrationVelocity,
                    SolverStabilizationHeuristicSettings = stabilizationSettings,
                    DirectSolverSettings = directSettings,
                    HaveStaticBodiesChanged = haveStaticBodiesChanged
                };

                if (Settings.synchronizeCollisionWorld == false)
                {
                    Debug.LogWarning("Synchronize Collision World is disabled, brick overlap may stale");
                }
            }

            debugFrameCount++;
            if (Settings.verboseLogging && debugFrameCount <= 10)
            {
                Debug.Log($"[SimStep {debugFrameCount}] CollisionWorld NumBodies: {physicsWorld.CollisionWorld.NumBodies}, NumDynamic: {physicsWorld.CollisionWorld.NumDynamicBodies}, NumStatic: {physicsWorld.CollisionWorld.NumStaticBodies}");
            }

            // Build the broadphase BVH trees before simulation
            using (s_PhysicsBuildBroadphaseMarker.Auto())
            {
                var buildBroadphaseHandle = physicsWorld.CollisionWorld.ScheduleBuildBroadphaseJobs(
                    ref physicsWorld, dt, Settings.gravity, haveStaticBodiesChanged, default, Settings.multiThreaded);
                using (s_PhysicsCompleteBroadphaseMarker.Auto())
                {
                    buildBroadphaseHandle.Complete();
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
                handles = simulation.ScheduleStepJobs(stepInput, default, Settings.multiThreaded);
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

            if (haveStaticBodiesChanged.Value > 0)
            {
                haveStaticBodiesChanged.Value = 0;
            }
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;

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

            if (Settings.verboseLogging) Debug.Log("Physics Disposed!");
        }
    }
}
