using Unity.Mathematics;
using Unity.Physics;
using UnityEngine;

namespace Caelix.Simulation
{
    /// <summary>
    /// Scene-authored physics settings for one world. This is a settings holder only: the
    /// simulation lives in <see cref="VoxelPhysicsWorld"/>, which is a plain class owned by the
    /// world. The host bootstrap reads <see cref="ToSettings"/> every frame, so inspector edits
    /// stay live-tunable while playing.
    /// </summary>
    /// <remarks>
    /// Field names match the pre-split <c>CaelixPhysicsWorld</c> component so a scene keeps its
    /// tuned values across the refactor.
    /// </remarks>
    public class PhysicsWorldConfig : MonoBehaviour
    {
        [Header("Gravity")]
        public Vector3 gravity = new Vector3(0, -9.81f, 0);

        [Header("Air Friction")]
        [Tooltip("Global linear velocity damping applied to every dynamic body each step (per-second, " +
                 "exponential). Higher = lower terminal falling/sliding speed. 0 = no linear air friction.")]
        [Min(0f)] public float linearAirFriction = 0.05f;

        [Tooltip("Global angular velocity damping applied to every dynamic body each step (per-second, " +
                 "exponential). 0 = no angular air friction.")]
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

        [Header("Contact Debug Logging")]
        [Tooltip("Master switch for voxel contact launch/instability logging.")]
        public bool enableContactDebugLogging = false;

        [Tooltip("Linear speed jump (m/s) within one step that counts as a launch and dumps the contact set.")]
        public float contactDebugSpikeSpeedDelta = 1.0f;

        [Header("Contact Profiling")]
        [Tooltip("Collect and log voxel narrowphase funnel counters. Needs the CAELIX_CONTACT_PROFILING scripting define.")]
        public bool enableContactProfiling = false;

        [Tooltip("Log every Nth step. 1 logs every step; larger values keep the console readable while the numbers settle.")]
        public int contactProfilingLogInterval = 60;

        [Tooltip("Average the counters over the interval instead of reporting only the last step.")]
        public bool contactProfilingAverage = true;

        public PhysicsWorldSettings ToSettings()
        {
            return new PhysicsWorldSettings
            {
                gravity = new float3(gravity.x, gravity.y, gravity.z),
                linearAirFriction = linearAirFriction,
                angularAirFriction = angularAirFriction,
                substepCount = substepCount,
                solverIterationCount = solverIterationCount,
                maxDynamicDepenetrationVelocity = maxDynamicDepenetrationVelocity,
                maxStaticDepenetrationVelocity = maxStaticDepenetrationVelocity,
                synchronizeCollisionWorld = synchronizeCollisionWorld,
                enableSolverStabilization = enableSolverStabilization,
                velocityClippingFactor = velocityClippingFactor,
                inertiaScalingFactor = inertiaScalingFactor,
                directSolverSettings = directSolverSettings,
                enableDirectSolver = enableDirectSolver,
                multiThreaded = multiThreaded,
                brickOverlapSerialThreshold = brickOverlapSerialThreshold,
                verboseLogging = verboseLogging,
                enableContactDebugLogging = enableContactDebugLogging,
                contactDebugSpikeSpeedDelta = contactDebugSpikeSpeedDelta,
                enableContactProfiling = enableContactProfiling,
                contactProfilingLogInterval = contactProfilingLogInterval,
                contactProfilingAverage = contactProfilingAverage,
            };
        }
    }
}
