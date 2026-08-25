using System.Threading;
using Unity.Burst;

namespace Unity.Physics
{
    /// <summary>
    /// Funnel counters for one narrowphase tick of voxel-vs-voxel contact generation.
    /// </summary>
    /// <remarks>
    /// Generation narrows in stages: body pairs give source features, each source feature sweeps a
    /// window of target roots, surviving roots give cells, and cells give contacts. Each field
    /// counts one stage, so the ratios between them say where the work goes and whether a stage is
    /// worth optimising. The two that decide whether the target loop should be restructured are
    /// WindowRoots / ContactsEmitted (how much of the window is wasted) and
    /// BrickCacheHits / BrickLookups (whether sector hash lookups still dominate).
    ///
    /// Instances are accumulated on the stack inside one body pair and flushed once, so the hot
    /// loops never touch shared memory. Without VOXELIS_CONTACT_PROFILING nothing reads them and
    /// the increments are dead stores that Burst removes.
    /// </remarks>
    public struct VoxelContactCounters
    {
        /// <summary>Voxel-vs-voxel body pairs that reached contact generation.</summary>
        public long BodyPairs;

        /// <summary>Source voxels that ran a vertex query.</summary>
        public long VertexSources;

        /// <summary>Source segments that ran an edge-edge query.</summary>
        public long EdgeSources;

        /// <summary>Target roots visited inside a query window, before any test.</summary>
        public long WindowRoots;

        /// <summary>Window roots that passed the occupancy bit.</summary>
        public long OccupiedRoots;

        /// <summary>Occupied roots that carried at least one usable cell.</summary>
        public long ActiveRoots;

        /// <summary>Closed-form distance evaluations, i.e. candidate feature pairs.</summary>
        public long CellTests;

        /// <summary>Brick resolution requests made by the queries.</summary>
        public long BrickLookups;

        /// <summary>Requests answered from the brick cache without a sector hash lookup.</summary>
        public long BrickCacheHits;

        /// <summary>Contacts appended to the raw list.</summary>
        public long ContactsEmitted;

        /// <summary>Cell tests dropped because the pair was further apart than maxDistance.</summary>
        public long ContactsOutOfRange;

        /// <summary>Contacts dropped because their witness shared a carrier and normal with one already reported.</summary>
        public long ContactsDeduped;

        /// <summary>Cell tests dropped because the two cores coincided, leaving the normal undefined.</summary>
        public long ContactsDegenerate;

        public void Add(in VoxelContactCounters other)
        {
            BodyPairs += other.BodyPairs;
            VertexSources += other.VertexSources;
            EdgeSources += other.EdgeSources;
            WindowRoots += other.WindowRoots;
            OccupiedRoots += other.OccupiedRoots;
            ActiveRoots += other.ActiveRoots;
            CellTests += other.CellTests;
            BrickLookups += other.BrickLookups;
            BrickCacheHits += other.BrickCacheHits;
            ContactsEmitted += other.ContactsEmitted;
            ContactsOutOfRange += other.ContactsOutOfRange;
            ContactsDeduped += other.ContactsDeduped;
            ContactsDegenerate += other.ContactsDegenerate;
        }
    }

    /// <summary>
    /// Collects <see cref="VoxelContactCounters"/> across every narrowphase job of a tick.
    /// </summary>
    /// <remarks>
    /// Compiled out unless VOXELIS_CONTACT_PROFILING is defined, so a normal build pays nothing:
    /// with no reader the per-root increments become dead stores. Add the define under
    /// Project Settings > Player > Scripting Define Symbols to measure, then arm it at runtime with
    /// <see cref="Enabled"/> - the define makes measurement possible, the flag chooses when.
    ///
    /// Narrowphase runs in parallel over body pairs, so the flush is one interlocked add per field
    /// per pair. Call <see cref="Reset"/> before scheduling a step and <see cref="Snapshot"/> after
    /// its jobs complete.
    /// </remarks>
    public static class VoxelContactProfiler
    {
#if VOXELIS_CONTACT_PROFILING
        // SharedStatic keys itself on type arguments, and a static class cannot be one, so the
        // owning context is a private placeholder rather than VoxelContactProfiler itself.
        private class ProfilerContext { }
        private class CountersKey { }
        private class EnabledKey { }

        private static readonly SharedStatic<VoxelContactCounters> s_Counters =
            SharedStatic<VoxelContactCounters>.GetOrCreate<ProfilerContext, CountersKey>();

        private static readonly SharedStatic<int> s_Enabled =
            SharedStatic<int>.GetOrCreate<ProfilerContext, EnabledKey>();
#endif

        /// <summary>
        /// Whether generation flushes its counters. Always false without VOXELIS_CONTACT_PROFILING.
        /// </summary>
        public static bool Enabled
        {
#if VOXELIS_CONTACT_PROFILING
            get => s_Enabled.Data != 0;
            set => s_Enabled.Data = value ? 1 : 0;
#else
            get => false;
            set { }
#endif
        }

        /// <summary>True when generation should bother accumulating. Readable from Burst.</summary>
        public static bool IsCollecting
        {
            get
            {
#if VOXELIS_CONTACT_PROFILING
                return s_Enabled.Data != 0;
#else
                return false;
#endif
            }
        }

        /// <summary>Clears the tick accumulator. Call before scheduling a step.</summary>
        public static void Reset()
        {
#if VOXELIS_CONTACT_PROFILING
            s_Counters.Data = default;
#endif
        }

        /// <summary>Totals for the tick. Call after the step's jobs have completed.</summary>
        public static VoxelContactCounters Snapshot()
        {
#if VOXELIS_CONTACT_PROFILING
            return s_Counters.Data;
#else
            return default;
#endif
        }

        /// <summary>
        /// Adds one body pair's stack counters into the tick accumulator. Called once per pair, so
        /// the interlocked adds stay off the per-root path.
        /// </summary>
        internal static void Flush(in VoxelContactCounters local)
        {
#if VOXELIS_CONTACT_PROFILING
            if (s_Enabled.Data == 0)
            {
                return;
            }

            ref VoxelContactCounters total = ref s_Counters.Data;
            Interlocked.Add(ref total.BodyPairs, local.BodyPairs);
            Interlocked.Add(ref total.VertexSources, local.VertexSources);
            Interlocked.Add(ref total.EdgeSources, local.EdgeSources);
            Interlocked.Add(ref total.WindowRoots, local.WindowRoots);
            Interlocked.Add(ref total.OccupiedRoots, local.OccupiedRoots);
            Interlocked.Add(ref total.ActiveRoots, local.ActiveRoots);
            Interlocked.Add(ref total.CellTests, local.CellTests);
            Interlocked.Add(ref total.BrickLookups, local.BrickLookups);
            Interlocked.Add(ref total.BrickCacheHits, local.BrickCacheHits);
            Interlocked.Add(ref total.ContactsEmitted, local.ContactsEmitted);
            Interlocked.Add(ref total.ContactsOutOfRange, local.ContactsOutOfRange);
            Interlocked.Add(ref total.ContactsDeduped, local.ContactsDeduped);
            Interlocked.Add(ref total.ContactsDegenerate, local.ContactsDegenerate);
#endif
        }
    }
}
