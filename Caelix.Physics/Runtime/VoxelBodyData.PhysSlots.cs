using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Caelix
{
    public partial struct VoxelBodyData : IDisposable
    {
        /// <summary>
        /// Recomputes the per-block <see cref="PhysicsInfo"/> slot of every brick that has pending
        /// require-update flags matching <paramref name="dirtyMask"/>. The slot encodes, for each
        /// solid block, the cells of the voxel-center cubical complex that are rooted at that block
        /// and are COLLISION-ACTIVE. The slot's aux bitmap is rebuilt in the same pass and marks the
        /// physics-key roots: a root is a key when it carries an active point or an active edge.
        /// Cross-brick topology is resolved through the 27-brick read window of the facade.
        /// </summary>
        /// <remarks>
        /// Selection uses the require-update (read) buffers populated by dirty propagation, mirroring
        /// the mesh renderer, plus the bricks whose slot storage was just created and therefore holds
        /// no valid derived data yet. The brick-level <c>GeometryWithLocalNeighbor</c> flag already
        /// covers bricks adjacent to a geometry change, so boundary blocks whose exposure flipped are
        /// re-evaluated too.
        ///
        /// The key rule is deliberately ROOT-LOCAL. The previous rule ("a root is a key when one of
        /// its cells covers a sparse seed voxel") was a two-step dependency - root, covered voxel,
        /// that voxel's own neighbor - which pushed the read window two voxels past the brick. The
        /// per-voxel propagation table in <c>NeighborhoodSettings.s_voxelPropagationMasks</c> only
        /// reaches ONE voxel, so a change at the neighbor brick's local index 1 set no direction bit
        /// towards this brick and this brick never refreshed. That silently froze key bits at the
        /// brick seam, and the harmful direction (clearing a voxel so a key should appear) dropped a
        /// contact source. A root-local rule keeps the window inside the one-voxel reach the
        /// propagation table guarantees, so the hole cannot recur.
        ///
        /// Every output is derived from block occupancy alone, never from another brick's
        /// <see cref="PhysicsInfo"/> or key bits. That matters because those are rewritten in place:
        /// a brick that is not selected this update still holds the previous update's ACTIVE bytes
        /// and key bits, so reading them as if they were raw occupancy would mix two representations.
        /// Occupancy is stable input for the whole refresh, so one parallel pass per brick is race
        /// free and deterministic.
        /// </remarks>
        private unsafe void RefreshPhysicsSlot(
            in VoxelEntityData entity,
            BrickUpdateFlags dirtyMask = BrickUpdateFlags.GeometryWithLocalNeighbor)
        {
            if (entity.RegionCount == 0)
            {
                return;
            }

            var keys = new NativeList<int3>(Allocator.TempJob);
            var required = new NativeList<RequiredBrick>(Allocator.TempJob);
            var unique = new NativeHashSet<int3>(64, Allocator.Temp);
            try
            {
                // Slot storage is created or grown on the main thread; the job only writes into
                // existing memory. The bricks reported here hold no valid derived data yet.
                entity.EnsureSlot<PhysicsInfo>(SectorSlotId.PhysicsInfo, BrickBitmask.Bytes, keys);
                entity.CollectRequiredBricks(default, dirtyMask, false, required);

                for (int i = 0; i < keys.Length; i++)
                {
                    unique.Add(keys[i]);
                }

                for (int i = 0; i < required.Length; i++)
                {
                    unique.Add(required[i].Key);
                }

                // The key list IS the selection; order does not matter.
                keys.Clear();
                foreach (int3 key in unique)
                {
                    keys.Add(key);
                }

                if (keys.Length == 0)
                {
                    return;
                }

                new ComputePhysicsSlotJob
                {
                    entity = entity,
                    keys = keys.AsArray()
                }.Schedule(keys.Length, 8).Complete();
            }
            finally
            {
                if (unique.IsCreated)
                {
                    unique.Dispose();
                }

                if (required.IsCreated)
                {
                    required.Dispose();
                }

                if (keys.IsCreated)
                {
                    keys.Dispose();
                }
            }
        }

        /// <summary>
        /// Burst job that fills the <see cref="PhysicsInfo"/> slot of one brick per index. Each index
        /// writes only into its own brick's slot data and aux (reads may cross into the 26 neighbor
        /// bricks), so running the selected bricks in parallel is data-race free.
        /// </summary>
        /// <remarks>
        /// Per brick the job first loads an occupancy window covering the brick plus a one-voxel halo
        /// on each side. The window is stored as one 10-bit row of X per (Y, Z) pair, so cell
        /// existence, activity and the key mask all become AND/OR/shift chains over eight voxels at a
        /// time. The window is filled from the Block slot's per-brick occupancy bitmask (rebuilt by
        /// <c>RefreshNonEmptyMask</c> immediately before this pass), which the facade binds for all
        /// 27 neighbors in one call instead of one lookup per neighbor test.
        /// </remarks>
        [BurstCompile]
        private unsafe struct ComputePhysicsSlotJob : IJobParallelFor
        {
            [ReadOnly] public VoxelEntityData entity;
            [ReadOnly] public NativeArray<int3> keys;

            // Window bounds in brick-local block coordinates. The low end reaches -1 because the cell
            // grown one voxel back is what a cell competes against for its negative directions. The
            // high end reaches +1 past the brick because a cube rooted at 7 spans the voxel at 8.
            //
            // ONE voxel each way is the whole reach, and that is load bearing: the per-voxel
            // propagation table only marks neighbor bricks within one voxel, so a window that reached
            // +2 would read voxels whose edits never flag this brick. Do not widen this without
            // widening s_voxelPropagationMasks to match.
            private const int WindowLow = -1;
            private const int WindowHigh = BrickKey.BlocksPerAxis;
            private const int WindowSpan = WindowHigh - WindowLow + 1;
            private const int WindowRows = WindowSpan * WindowSpan;

            // Highest root coordinate.
            private const int RootHigh = BrickKey.BlocksPerAxis - 1;

            // Bit b of a window row holds the occupancy of x = b - 1.
            private const int RowBitOrigin = -WindowLow;

            private const int NeighborBrickCount = 27;

            public unsafe void Execute(int index)
            {
                int3 key = keys[index];
                if (!entity.TryBindBrick<PhysicsInfo>(SectorSlotId.PhysicsInfo, key, out PhysicsInfo* physBrick) ||
                    !entity.TryBindBrickAux(SectorSlotId.PhysicsInfo, key, out void* keyMask))
                {
                    return;
                }

                // Scratch for one brick.
                ulong** brickMasks = stackalloc ulong*[NeighborBrickCount];
                Block** brickBlocks = stackalloc Block*[NeighborBrickCount];
                uint* occupancyRows = stackalloc uint[WindowRows];
                uint* cellRows = stackalloc uint[PhysicsInfo.FeatureBitCount * WindowRows];
                uint* activeRows = stackalloc uint[PhysicsInfo.FeatureBitCount];

                VoxelNeighborhood window = entity.OpenNeighborhood(key, SectorSlotId.Block);
                LoadNeighborBricks(ref window, brickMasks, brickBlocks);
                LoadOccupancyWindow(brickMasks, brickBlocks, occupancyRows);
                ComputeCellRows(occupancyRows, cellRows);
                WriteBrick(cellRows, activeRows, physBrick, (ulong*)keyMask);
            }

            /// <summary>Row index of one X row of the occupancy window.</summary>
            private static int RowIndex(int y, int z)
            {
                return (z - WindowLow) * WindowSpan + (y - WindowLow);
            }

            /// <summary>
            /// Caches the 3x3x3 bricks the window spans. Absent bricks stay null and read as empty.
            /// The occupancy bitmask is preferred; the raw blocks are the fallback for a brick whose
            /// Block slot carries no aux yet.
            /// </summary>
            private static unsafe void LoadNeighborBricks(
                ref VoxelNeighborhood window,
                ulong** brickMasks,
                Block** brickBlocks)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int slot = VoxelNeighborhood.Index(dx, dy, dz);
                            brickMasks[slot] = (ulong*)window.GetAuxPtr(slot);
                            brickBlocks[slot] = window.GetBrickPtr(slot);
                        }
                    }
                }
            }

            /// <summary>Occupancy of one in-brick X row, as eight bits with x at bit x.</summary>
            private static unsafe uint BrickRowBits(
                ulong** brickMasks, Block** brickBlocks, int slot, int y, int z)
            {
                ulong* mask = brickMasks[slot];
                if (mask != null)
                {
                    // One mask word covers one Z slice; within it a row starts at y * BlocksPerAxis.
                    return (uint)((mask[z] >> (y << BrickKey.Shift)) & 0xFFul);
                }

                Block* blocks = brickBlocks[slot];
                if (blocks == null)
                {
                    return 0u;
                }

                int baseIdx = BrickKey.ToBlockIdx(0, y, z);
                uint bits = 0u;
                for (int x = 0; x < BrickKey.BlocksPerAxis; x++)
                {
                    bits |= blocks[baseIdx + x].isEmpty ? 0u : (1u << x);
                }
                return bits;
            }

            /// <summary>
            /// Fills the occupancy window. Each row splices the last bit of the -X brick, all eight
            /// bits of the centre brick and the first bit of the +X brick. The +X brick's remaining
            /// bits land above the window and are simply unused; they hold correct occupancy, so a
            /// shift that reaches them cannot read stale data.
            /// </summary>
            private static unsafe void LoadOccupancyWindow(
                ulong** brickMasks, Block** brickBlocks, uint* occupancyRows)
            {
                for (int z = WindowLow; z <= WindowHigh; z++)
                {
                    int brickZ = z >> BrickKey.Shift;
                    int localZ = z & BrickKey.Mask;
                    for (int y = WindowLow; y <= WindowHigh; y++)
                    {
                        int brickY = y >> BrickKey.Shift;
                        int localY = y & BrickKey.Mask;

                        int centre = VoxelNeighborhood.Index(0, brickY, brickZ);
                        uint low = BrickRowBits(brickMasks, brickBlocks, centre - 1, localY, localZ);
                        uint mid = BrickRowBits(brickMasks, brickBlocks, centre, localY, localZ);
                        uint high = BrickRowBits(brickMasks, brickBlocks, centre + 1, localY, localZ);

                        occupancyRows[RowIndex(y, z)] =
                            (low >> (BrickKey.BlocksPerAxis - RowBitOrigin)) |
                            (mid << RowBitOrigin) |
                            (high << (BrickKey.BlocksPerAxis + RowBitOrigin));
                    }
                }
            }

            /// <summary>
            /// Step 1: existence of every cell of the complex, indexed by axis mask (X=1, Y=2, Z=4).
            /// A cell exists when all of its corner voxels are occupied, so each mask is an AND of
            /// the rows and shifts its axes select. Rows are produced for every root a brick voxel
            /// competes against, which includes one row back on Y and Z.
            /// </summary>
            private static unsafe void ComputeCellRows(uint* occupancyRows, uint* cellRows)
            {
                for (int z = WindowLow; z <= RootHigh; z++)
                {
                    for (int y = WindowLow; y <= RootHigh; y++)
                    {
                        int row = RowIndex(y, z);
                        uint here = occupancyRows[row];
                        uint aheadY = occupancyRows[RowIndex(y + 1, z)];
                        uint aheadZ = occupancyRows[RowIndex(y, z + 1)];
                        uint aheadYZ = occupancyRows[RowIndex(y + 1, z + 1)];

                        uint spanX = here & (here >> 1);
                        uint spanXaheadY = aheadY & (aheadY >> 1);
                        uint spanXaheadZ = aheadZ & (aheadZ >> 1);
                        uint spanYZ = here & aheadY & aheadZ & aheadYZ;

                        cellRows[0 * WindowRows + row] = here;                   // point
                        cellRows[1 * WindowRows + row] = spanX;                  // X
                        cellRows[2 * WindowRows + row] = here & aheadY;          // Y
                        cellRows[3 * WindowRows + row] = spanX & spanXaheadY;    // XY
                        cellRows[4 * WindowRows + row] = here & aheadZ;          // Z
                        cellRows[5 * WindowRows + row] = spanX & spanXaheadZ;    // XZ
                        cellRows[6 * WindowRows + row] = spanYZ;                 // YZ
                        cellRows[7 * WindowRows + row] = spanYZ & (spanYZ >> 1); // XYZ
                    }
                }
            }

            /// <summary>
            /// Step 2: keeps only the cells that are collision-active. A cell that can grow along
            /// axis <c>a</c> gives every direction with a positive <c>+a</c> component to the grown
            /// cell rooted here, and every direction with a positive <c>-a</c> component to the grown
            /// cell rooted one voxel back. Both together leave only a zero-area slice, so the cell
            /// survives exactly when at least one of the two grown cells is missing.
            /// <para>
            /// This is the old containment rule with AND in place of OR: dedup dropped a cell when
            /// EITHER grown cell existed, activity drops it only when BOTH do. The cube has no axis
            /// to grow along, so it is always active; it is a volume cell and the surface path
            /// excludes it. Results are indexed by <see cref="PhysicsInfo"/> feature bit.
            /// </para>
            /// </summary>
            private static unsafe void ComputeActiveRows(
                uint* cellRows, int y, int z, uint* activeRows)
            {
                int row = RowIndex(y, z);
                int rowBackY = RowIndex(y - 1, z);
                int rowBackZ = RowIndex(y, z - 1);

                for (int axisMask = 0; axisMask < PhysicsInfo.FeatureBitCount; axisMask++)
                {
                    uint covered = 0u;
                    if ((axisMask & 1) == 0)
                    {
                        // One voxel back on X is one bit up in the row.
                        uint grownX = cellRows[(axisMask | 1) * WindowRows + row];
                        covered |= grownX & (grownX << 1);
                    }
                    if ((axisMask & 2) == 0)
                    {
                        covered |= cellRows[(axisMask | 2) * WindowRows + row] &
                                   cellRows[(axisMask | 2) * WindowRows + rowBackY];
                    }
                    if ((axisMask & 4) == 0)
                    {
                        covered |= cellRows[(axisMask | 4) * WindowRows + row] &
                                   cellRows[(axisMask | 4) * WindowRows + rowBackZ];
                    }

                    activeRows[PhysicsInfo.FeatureBitFromAxisMask(axisMask)] =
                        cellRows[axisMask * WindowRows + row] & ~covered;
                }
            }

            /// <summary>
            /// Step 3: a root is a key when it carries an active point or an active edge. Every
            /// permitted feature pair (vertex-vertex, vertex-edge, vertex-face, edge-edge) has a
            /// vertex or an edge on at least one side, so these roots are exactly the contact
            /// sources; a face-only root is always the target of a vertex.
            /// <para>
            /// Root-local by design. Deriving the key from cells covering a neighboring seed voxel
            /// would reach two voxels past the brick, which is one more than dirty propagation
            /// guarantees.
            /// </para>
            /// </summary>
            private static unsafe uint ComputeKeyRow(uint* activeRows)
            {
                return activeRows[PhysicsInfo.BitPoint] |
                       activeRows[PhysicsInfo.BitEdgeX] |
                       activeRows[PhysicsInfo.BitEdgeY] |
                       activeRows[PhysicsInfo.BitEdgeZ];
            }

            /// <summary>
            /// Transposes the per-row results into one byte per voxel and one mask word per Z slice.
            /// Air voxels fall out as zero because no cell exists at an empty root.
            /// </summary>
            private static unsafe void WriteBrick(
                uint* cellRows,
                uint* activeRows,
                PhysicsInfo* physBrick,
                ulong* physicsKeyMask)
            {
                for (int z = 0; z < BrickKey.BlocksPerAxis; z++)
                {
                    ulong keyWord = 0ul;
                    for (int y = 0; y < BrickKey.BlocksPerAxis; y++)
                    {
                        ComputeActiveRows(cellRows, y, z, activeRows);
                        uint keyRow = ComputeKeyRow(activeRows);

                        int baseIdx = BrickKey.ToBlockIdx(0, y, z);
                        for (int x = 0; x < BrickKey.BlocksPerAxis; x++)
                        {
                            int bit = x + RowBitOrigin;
                            uint data = 0u;
                            for (int feature = 0; feature < PhysicsInfo.FeatureBitCount; feature++)
                            {
                                data |= ((activeRows[feature] >> bit) & 1u) << feature;
                            }
                            physBrick[baseIdx + x] = new PhysicsInfo { data = (byte)data };
                        }

                        keyWord |= (ulong)((keyRow >> RowBitOrigin) & 0xFFu) <<
                                   (y << BrickKey.Shift);
                    }
                    physicsKeyMask[z] = keyWord;
                }
            }
        }
    }
}
