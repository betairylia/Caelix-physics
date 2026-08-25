using System.Runtime.InteropServices;
using System.Threading;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Funnel counters for one contact query, i.e. one of the two ways generation finds targets.
    /// </summary>
    /// <remarks>
    /// The two queries load very differently by scene: large bodies resting on ground are rim-heavy
    /// and run mostly edge sources, while machinery made of cogs and chains is corner-heavy and runs
    /// mostly vertex sources. A blended total hides which one a change actually moved, so each query
    /// carries its own copy.
    ///
    /// Stages narrow in order: sources sweep window roots, occupied roots survive the occupancy
    /// mask, active roots carry a usable cell, cells become distance tests, tests become contacts.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct VoxelContactQueryCounters
    {
        /// <summary>Source features that ran this query.</summary>
        public long Sources;

        /// <summary>Target roots covered by a query window inside a resolved brick.</summary>
        /// <remarks>
        /// Counted arithmetically from the window extent, not by iterating, so it still measures the
        /// sweep the query is responsible for after the occupancy mask stopped it from touching
        /// empty voxels. WindowRoots against TouchedRoots is therefore the work avoided.
        /// </remarks>
        public long WindowRoots;

        /// <summary>
        /// Window roots the bitmask prefilter admitted, i.e. the only ones actually read.
        /// </summary>
        /// <remarks>
        /// The prefilter differs by query. The vertex query scans the occupancy mask, because any
        /// occupied root can carry a cell it may pair with. The edge query scans the physics key
        /// mask instead: a root with an active edge is a contact source by definition, so the key
        /// mask is a strict superset of its targets and a far sparser one than occupancy.
        /// </remarks>
        public long TouchedRoots;

        /// <summary>Touched roots carrying at least one cell usable by this query.</summary>
        public long ActiveRoots;

        /// <summary>Closed-form distance evaluations, i.e. candidate feature pairs.</summary>
        public long CellTests;

        /// <summary>Single-word occupancy tests, one per voxel row of a window.</summary>
        public long RowsTested;

        /// <summary>Rows a single word test rejected whole, skipping every voxel in them.</summary>
        public long RowsSkipped;

        /// <summary>Brick resolution requests.</summary>
        public long BrickLookups;

        /// <summary>
        /// Sector lookups actually performed, i.e. the requests the brick cache could not answer.
        /// This is the cost brick resolution really carries; BrickLookups counts requests, most of
        /// which the cache answers for free.
        /// </summary>
        public long BrickResolves;

        /// <summary>Requests answered from the brick cache without a sector hash lookup.</summary>
        public long BrickCacheHits;

        /// <summary>Contacts appended to the raw list.</summary>
        public long ContactsEmitted;

        /// <summary>Cell tests dropped because the pair was further apart than maxDistance.</summary>
        public long ContactsOutOfRange;

        /// <summary>Contacts dropped for sharing a carrier and normal with one already reported.</summary>
        public long ContactsDeduped;

        /// <summary>Cell tests dropped because the cores coincided, leaving the normal undefined.</summary>
        public long ContactsDegenerate;

        public void Add(in VoxelContactQueryCounters other)
        {
            Sources += other.Sources;
            WindowRoots += other.WindowRoots;
            TouchedRoots += other.TouchedRoots;
            ActiveRoots += other.ActiveRoots;
            CellTests += other.CellTests;
            RowsTested += other.RowsTested;
            RowsSkipped += other.RowsSkipped;
            BrickLookups += other.BrickLookups;
            BrickResolves += other.BrickResolves;
            BrickCacheHits += other.BrickCacheHits;
            ContactsEmitted += other.ContactsEmitted;
            ContactsOutOfRange += other.ContactsOutOfRange;
            ContactsDeduped += other.ContactsDeduped;
            ContactsDegenerate += other.ContactsDegenerate;
        }
    }

    /// <summary>
    /// Funnel counters for one narrowphase tick of voxel-vs-voxel contact generation.
    /// </summary>
    /// <remarks>
    /// Accumulated on the stack inside one body pair and flushed once, so the hot loops never touch
    /// shared memory. Without VOXELIS_CONTACT_PROFILING nothing reads them and the increments are
    /// dead stores that Burst removes.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct VoxelContactCounters
    {
        /// <summary>Voxel-vs-voxel body pairs that reached contact generation.</summary>
        public long BodyPairs;

        /// <summary>Source bricks whose key voxels were enumerated.</summary>
        public long SourceBricks;

        /// <summary>The vertex query: source is an active vertex, targets are all active cells.</summary>
        public VoxelContactQueryCounters Vertex;

        /// <summary>The edge-edge query: source is an active edge, targets are active edges.</summary>
        public VoxelContactQueryCounters Edge;

        /// <summary>Both queries summed, for the scene-level view.</summary>
        public VoxelContactQueryCounters Total
        {
            get
            {
                VoxelContactQueryCounters total = Vertex;
                total.Add(Edge);
                return total;
            }
        }

        public void Add(in VoxelContactCounters other)
        {
            BodyPairs += other.BodyPairs;
            SourceBricks += other.SourceBricks;
            Vertex.Add(other.Vertex);
            Edge.Add(other.Edge);
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
    ///
    /// Counters are held in a fixed-size slot buffer rather than in the counter struct itself,
    /// because Burst's shared-static registry is native and outlives domain reloads: a payload that
    /// changes size throws on every access until the Editor process restarts.
    /// </remarks>
    public static unsafe class VoxelContactProfiler
    {
#if VOXELIS_CONTACT_PROFILING
        // Burst registers a SharedStatic in a NATIVE registry keyed by type, and that registry
        // survives domain reloads - only a process restart clears it. If the payload's size ever
        // changes, every access throws until the Editor is restarted. So the payload is a fixed
        // slot buffer whose size never moves, and the counters are copied through it as a flat run
        // of longs. Adding a counter above stays a recompile, not a restart.
        private const int k_SlotCount = 64;

        [StructLayout(LayoutKind.Sequential)]
        private struct CounterSlots
        {
            public fixed long Values[k_SlotCount];
        }

        // SharedStatic keys itself on type arguments, and a static class cannot be one, so the
        // owning context is a private placeholder rather than VoxelContactProfiler itself.
        private class ProfilerContext { }
        private class CountersKey { }
        private class EnabledKey { }

        private static readonly SharedStatic<CounterSlots> s_Slots =
            SharedStatic<CounterSlots>.GetOrCreate<ProfilerContext, CountersKey>();

        private static readonly SharedStatic<int> s_Enabled =
            SharedStatic<int>.GetOrCreate<ProfilerContext, EnabledKey>();

        private static long* SlotPtr => (long*)UnsafeUtility.AddressOf(ref s_Slots.Data);

        /// <summary>Longs in <see cref="VoxelContactCounters"/>, which is a flat run of them.</summary>
        private static int SlotsUsed => UnsafeUtility.SizeOf<VoxelContactCounters>() / sizeof(long);
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
            // Main thread only, so this is where an oversized counter struct is caught: the flush
            // runs inside Burst jobs and cannot report anything.
            if (SlotsUsed > k_SlotCount)
            {
                throw new System.InvalidOperationException(
                    "VoxelContactCounters needs " + SlotsUsed + " slots but only "
                    + k_SlotCount + " exist. Raise k_SlotCount.");
            }

            long* slots = SlotPtr;
            for (int i = 0; i < k_SlotCount; i++)
            {
                slots[i] = 0L;
            }
#endif
        }

        /// <summary>Totals for the tick. Call after the step's jobs have completed.</summary>
        public static VoxelContactCounters Snapshot()
        {
#if VOXELIS_CONTACT_PROFILING
            VoxelContactCounters result = default;
            long* destination = (long*)UnsafeUtility.AddressOf(ref result);
            long* slots = SlotPtr;
            int used = SlotsUsed;
            for (int i = 0; i < used && i < k_SlotCount; i++)
            {
                destination[i] = slots[i];
            }
            return result;
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

            // Every field is a long, so the whole struct folds in with one loop. That keeps the
            // flush correct when counters are added or reordered, with no per-field list to update.
            VoxelContactCounters source = local;
            long* values = (long*)UnsafeUtility.AddressOf(ref source);
            long* slots = SlotPtr;
            int used = SlotsUsed;
            for (int i = 0; i < used && i < k_SlotCount; i++)
            {
                Interlocked.Add(ref slots[i], values[i]);
            }
#endif
        }
    }
}
