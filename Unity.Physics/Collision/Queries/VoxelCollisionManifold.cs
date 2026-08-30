#define SHOW_DEBUG

using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Voxelis;
using static Unity.Physics.Math;



#if SHOW_DEBUG
using UnityEngine;
#endif

namespace Unity.Physics
{
    static partial class ManifoldQueries
    {
        // ---------------------------------------------------------------------------------------
        // Voxel vs voxel contact generation.
        //
        // Occupied voxel centers form a finite cubical complex. Two adjacent centers span a
        // segment, four span a square, eight span a cube. The collision surface is that complex
        // swept by a sphere of radius 0.5. A cell is written (r, m): r is a voxel coordinate, m is
        // an axis mask (X=1, Y=2, Z=4), and the cell spans [r, r+m] in voxel-center space.
        //
        // PhysicsInfo.data marks which cells rooted at a voxel are collision-active. Contact
        // generation dispatches exactly the feature pairs whose dimensions sum to at most two:
        // vertex-vertex, vertex-edge, vertex-face and edge-edge. Every one of those has a vertex on
        // at least one side except edge-edge, so generation splits into two queries:
        //
        //   VertexQuery    source is an active vertex, targets are all active cells.
        //   EdgeEdgeQuery  source is an active edge, targets are active edges.
        //
        // Faces are never sources. A face only ever meets a vertex, and that vertex is a source in
        // its own body's pass.
        //
        // Both queries locate their targets with a window of target voxel roots sized so that every
        // cell within reach is enumerated (ComputeTargetRootWindow). Target cells are axis-aligned
        // boxes in target grid space, so every dispatched pair has a closed form: a clamp for the
        // vertex pairs and a clamped segment-segment solve for edge-edge. Nothing here calls a
        // general convex-distance query.
        //
        // Witness points on a shared boundary are canonicalized to their carrier - the
        // lowest-dimensional cell containing them - so the same geometric contact found through
        // several touching cells is emitted once.
        //
        // This file is GENERATION ONLY. Every surviving pair becomes one raw contact, written as
        // its own single-point manifold plus one contact event. There is no patch merging or
        // reduction, and containment / deep overlap are not covered.
        //
        // Conventions, matching the rest of the narrowphase:
        //   * Manifold normals point from B towards A; positive contact distance is separation.
        //   * Positions are in world space, on body B's surface.
        // Uniform body scale is assumed to be 1.
        //
        // Design, derivations and history: VoxelisX/VOXEL_CONTACT_GENERATION.md
        // ---------------------------------------------------------------------------------------

        // Cap on the speculative margin used to size the candidate window. The contact distance
        // gate still uses the full maxDistance; this only bounds the search volume.
        const float k_VoxelMaxSpeculativeMargin = 0.5f;

        // Radius swept over the cell complex, on both bodies.
        const float k_VoxelCoreRadius = 0.5f;

        // Dispatched pairs are exactly those with dim(a) + dim(b) <= 2.
        const int k_MaxPermittedDimensionSum = 2;

        // Distance from a voxel-center plane below which a witness coordinate counts as lying on
        // that plane, i.e. its carrier does not span that axis.
        const float k_CarrierSnapTolerance = 1e-4f;

        // Below this core distance the contact normal is undefined and the pair is dropped.
        const float k_MinCoreDistanceSq = 1e-8f;

        // Two contact normals closer than this in dot product count as the same direction when
        // deciding whether two witnesses on one carrier are the same contact.
        const float k_SameNormalDot = 0.999999f;

        // One raw generated contact, in B grid space.
        struct VoxelContact
        {
            public float Distance;      // separation along the normal, negative when penetrating
            public float3 NormalInB;    // unit normal, points from B towards A
            public float3 PosInB;       // contact position on B's surface
            public int3 VoxelA;         // contributing A voxel, A grid
            public int3 VoxelB;         // contributing B voxel, B grid
            public int3 CarrierRoot;    // root of the lowest-dimensional cell holding the witness
            public byte CarrierMask;    // axis mask of that cell
            public byte TargetAxisMask; // axis mask of the target cell that produced the contact
        }

        unsafe struct VoxelBrickView
        {
            public PhysicsInfo* Physics;
            public ulong* OccupiedMask;
            public ulong* PhysicsKeyMask;
        }

        // Direct-mapped cache of resolved bricks, indexed by the low bits of the brick coordinate
        // rather than by a hash.
        //
        // Every key voxel sweeps a window a few voxels wide and requests two to four bricks to read
        // a handful of voxels, so without this each one repeats a sector hash lookup. Neighbouring
        // source voxels - and neighbouring source bricks - request overlapping sets, and indexing
        // by position keeps those in the same slots, so the reuse spans the whole pass instead of
        // being lost whenever a hash collides or the addressing shifts.
        //
        // Two bricks alias only when a coordinate differs by a multiple of four, which is outside
        // the few-brick reach of any one source voxel. A stale entry is caught by the coordinate
        // check, so aliasing costs a resolve and never correctness.
        struct BrickCacheEntry
        {
            public int3 Coord;
            public VoxelBrickView View;
            public bool Valid;
            public bool Present;
        }

        // Four bricks per axis, so the index is three pairs of bits and the table stays small
        // enough to clear cheaply once per pass.
        const int k_BrickCacheAxisMask = 3;
        const int k_BrickCacheSize = 64;

        internal static unsafe void VoxelVoxel(
            Context context,
            Collider* colliderA,
            Collider* colliderB,
            [NoAlias] in MTransform worldFromA,
            [NoAlias] in MTransform worldFromB,
            float maxDistance,
            bool flipped)
        {
            VoxelCollider* voxelA = (VoxelCollider*)colliderA;
            VoxelCollider* voxelB = (VoxelCollider*)colliderB;

            Material materialA = voxelA->Material;
            Material materialB = voxelB->Material;

            CollisionResponsePolicy combinedCollisionResponse =
                Material.GetCombinedCollisionResponse(materialA, materialB);

            if (combinedCollisionResponse == CollisionResponsePolicy.None)
            {
                return;
            }

            // Infinite mass on both sides: only worth running when a material wants solver events.
            if (context.BothMotionsAreKinematic)
            {
                if (combinedCollisionResponse != CollisionResponsePolicy.RaiseTriggerEvents &&
                    combinedCollisionResponse != CollisionResponsePolicy.CollideRaiseCollisionEvents)
                {
                    return;
                }
            }

            if (!voxelA->m_Sectors.IsCreated || !voxelB->m_Sectors.IsCreated ||
                voxelA->m_Sectors.IsEmpty || voxelB->m_Sectors.IsEmpty)
            {
                return;
            }

            // The A-side pass walks A's sectors, so put the smaller body there.
            // TODO: Make this more precise perhaps
            bool isALargerThanB = (voxelA->m_Sectors.Count > voxelB->m_Sectors.Count);
            if (isALargerThanB)
            {
                _VoxelVoxel(
                    context, voxelB, voxelA, materialB, materialA,
                    worldFromB, worldFromA, maxDistance, !flipped);
                return;
            }

            _VoxelVoxel(
                context, voxelA, voxelB, materialA, materialB,
                worldFromA, worldFromB, maxDistance, flipped);
        }

        private static unsafe void _VoxelVoxel(
            Context context,
            VoxelCollider* voxelA,
            VoxelCollider* voxelB,
            Material materialA,
            Material materialB,
            [NoAlias] in MTransform worldFromA,
            [NoAlias] in MTransform worldFromB,
            float maxDistance,
            bool flipped)
        {
            MTransform bFromA = Mul(Inverse(worldFromB), worldFromA);

            var contacts = new UnsafeList<VoxelContact>(256, Allocator.Temp);

            // Accumulated on the stack for this body pair and flushed once, so the per-root loops
            // never touch shared memory. See VoxelContactProfiler.
            var counters = new VoxelContactCounters { BodyPairs = 1 };

            CollectVoxelContacts(voxelA, voxelB, bFromA, maxDistance, ref counters, ref contacts);
            WriteVoxelManifolds(contacts, context, worldFromB, materialA, materialB, flipped);
            contacts.Dispose();

            VoxelContactProfiler.Flush(counters);
        }

        // -----------------------------------------------------------------------------------
        // Target window
        // -----------------------------------------------------------------------------------

        // Roots of every target cell that can lie within `reach` of a source feature bounded by
        // [sourceMinimum, sourceMaximum], both in target grid space.
        //
        // Let W_box be the box spanned by the window's voxel CENTERS. A cell that misses W_box is
        // separated from it on some axis, and requiring that gap to exceed the reach gives
        //   lo < sourceMinimum - reach + 1     and     hi > sourceMaximum + reach - 1
        // measured in centers, hence the half-voxel shifts below. A cell meeting W_box has its
        // root no lower than lo - 1, because a cell that spans an axis is rooted one voxel back
        // from where it reaches, so the enumerated root range starts one voxel earlier.
        //
        // For a point source this reduces to the 2x2x2 bracket of voxels around the point, widened
        // by one voxel on any axis where the point lies within the margin of the bracket face.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ComputeTargetRootWindow(
            float3 sourceMinimum,
            float3 sourceMaximum,
            float reach,
            out int3 lower,
            out int3 upper)
        {
            lower = (int3)math.ceil(sourceMinimum - reach + 0.5f) - 2;
            upper = (int3)math.floor(sourceMaximum + reach - 1.5f) + 1;
        }

        // Axis-aligned box of a target cell, in target grid space.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void GetCellBox(int3 root, int axisMask, out float3 minimum, out float3 maximum)
        {
            minimum = (float3)root + 0.5f;
            maximum = minimum + new float3(
                (axisMask & 1) != 0 ? 1.0f : 0.0f,
                (axisMask & 2) != 0 ? 1.0f : 0.0f,
                (axisMask & 4) != 0 ? 1.0f : 0.0f);
        }

        // -----------------------------------------------------------------------------------
        // Closed-form distances
        // -----------------------------------------------------------------------------------

        // Closest points between two segments, each given as origin plus full edge vector and
        // parameterised over [0, 1].
        private static void ClosestPointsSegmentSegment(
            float3 originP, float3 edgeP,
            float3 originQ, float3 edgeQ,
            out float3 closestOnP, out float3 closestOnQ)
        {
            const float degenerate = 1e-9f;

            float3 between = originP - originQ;
            float lengthSqP = math.lengthsq(edgeP);
            float lengthSqQ = math.lengthsq(edgeQ);
            float projectedQ = math.dot(edgeQ, between);

            float s;
            float t;
            if (lengthSqP <= degenerate && lengthSqQ <= degenerate)
            {
                s = 0.0f;
                t = 0.0f;
            }
            else if (lengthSqP <= degenerate)
            {
                s = 0.0f;
                t = math.saturate(projectedQ / lengthSqQ);
            }
            else
            {
                float projectedP = math.dot(edgeP, between);
                if (lengthSqQ <= degenerate)
                {
                    t = 0.0f;
                    s = math.saturate(-projectedP / lengthSqP);
                }
                else
                {
                    float betweenEdges = math.dot(edgeP, edgeQ);
                    float denominator = lengthSqP * lengthSqQ - betweenEdges * betweenEdges;

                    // Parallel segments leave s free; anchoring it at 0 and clamping t below
                    // still lands on a closest pair.
                    s = denominator > degenerate
                        ? math.saturate((betweenEdges * projectedQ - projectedP * lengthSqQ) / denominator)
                        : 0.0f;

                    t = (betweenEdges * s + projectedQ) / lengthSqQ;
                    if (t < 0.0f)
                    {
                        t = 0.0f;
                        s = math.saturate(-projectedP / lengthSqP);
                    }
                    else if (t > 1.0f)
                    {
                        t = 1.0f;
                        s = math.saturate((betweenEdges - projectedP) / lengthSqP);
                    }
                }
            }

            closestOnP = originP + s * edgeP;
            closestOnQ = originQ + t * edgeQ;
        }

        // -----------------------------------------------------------------------------------
        // Carrier and emission
        // -----------------------------------------------------------------------------------

        // Lowest-dimensional cell of the complex containing `witness`, in target grid space. A
        // witness coordinate sitting on a voxel-center plane means the carrier does not span that
        // axis; otherwise it lies strictly between two centers and the carrier spans it.
        private static void GetWitnessCarrier(float3 witness, out int3 root, out byte axisMask)
        {
            float3 onLattice = witness - 0.5f;
            float3 nearest = math.round(onLattice);
            bool3 onPlane = math.abs(onLattice - nearest) < k_CarrierSnapTolerance;

            root = (int3)math.select(math.floor(onLattice), nearest, onPlane);
            axisMask = (byte)((onPlane.x ? 0 : 1) | (onPlane.y ? 0 : 2) | (onPlane.z ? 0 : 4));
        }

        // Appends one contact for a pair of core witness points, both already in B grid space.
        // `recentCount` is how many contacts the current source feature has already appended; a
        // witness that canonicalizes onto a carrier already reported with the same normal is the
        // same geometric contact reached through another touching cell, so it is dropped.
        private static unsafe bool EmitCoreContact(
            float3 corePointAinB,
            float3 corePointBinB,
            int3 voxelA,
            int3 voxelB,
            byte targetAxisMask,
            float maxDistance,
            int recentCount,
            ref VoxelContactQueryCounters counters,
            ref UnsafeList<VoxelContact> contacts)
        {
            float3 delta = corePointAinB - corePointBinB;
            float coreDistanceSq = math.lengthsq(delta);
            if (coreDistanceSq <= k_MinCoreDistanceSq)
            {
                counters.ContactsDegenerate++;
                return false;
            }

            float coreDistance = math.sqrt(coreDistanceSq);
            float distance = coreDistance - 2.0f * k_VoxelCoreRadius;
            if (distance >= maxDistance || !math.isfinite(distance))
            {
                counters.ContactsOutOfRange++;
                return false;
            }

            float3 normalInB = delta / coreDistance;
            if (!math.all(math.isfinite(normalInB)))
            {
                return false;
            }

            GetWitnessCarrier(corePointBinB, out int3 carrierRoot, out byte carrierMask);

            // TODO: FIXME: This looks sus?
            for (int i = contacts.Length - recentCount; i < contacts.Length; i++)
            {
                VoxelContact existing = contacts[i];
                if (existing.CarrierMask == carrierMask &&
                    math.all(existing.CarrierRoot == carrierRoot) &&
                    math.dot(existing.NormalInB, normalInB) > k_SameNormalDot)
                {
                    counters.ContactsDeduped++;
                    return false;
                }
            }

            contacts.Add(new VoxelContact
            {
                Distance = distance,
                NormalInB = normalInB,
                PosInB = corePointBinB + normalInB * k_VoxelCoreRadius,
                VoxelA = voxelA,
                VoxelB = voxelB,
                CarrierRoot = carrierRoot,
                CarrierMask = carrierMask,
                TargetAxisMask = targetAxisMask
            });
            counters.ContactsEmitted++;
            return true;
        }

        // -----------------------------------------------------------------------------------
        // Brick resolution
        // -----------------------------------------------------------------------------------

        private static unsafe bool ResolveVoxelBrick(
            VoxelCollider* collider,
            int3 globalBrickCoord,
            out VoxelBrickView view)
        {
            int3 sectorCoord = globalBrickCoord >> Sector.SHIFT_IN_BRICKS;
            if (!collider->m_Sectors.TryGetValue(sectorCoord, out SectorHandle handle) || handle.IsNull)
            {
                view = default;
                return false;
            }

            int3 brickInSector = globalBrickCoord & Sector.SECTOR_MASK;
            Sector* sector = handle.Ptr;
            short bid = sector->brickIdx[
                Sector.ToBrickIdx(brickInSector.x, brickInSector.y, brickInSector.z)];
            if (bid == Sector.BRICKID_EMPTY)
            {
                view = default;
                return false;
            }

            SectorSlotStorage* blockSlot = sector->slots + (int)SectorSlotId.Block;
            SectorSlotStorage* physicsSlot = sector->slots + (int)SectorSlotId.PhysicsInfo;

            // Production rebuilds both aux masks before physics. Without either mask this brick
            // cannot use the sparse contact path safely.
            if (!blockSlot->IsCreated || !blockSlot->HasAux ||
                !physicsSlot->IsCreated || !physicsSlot->HasAux)
            {
                view = default;
                return false;
            }

            view = new VoxelBrickView
            {
                Physics = (PhysicsInfo*)physicsSlot->GetBrickPtr(bid),
                OccupiedMask = (ulong*)blockSlot->GetBrickAuxPtr(bid),
                PhysicsKeyMask = (ulong*)physicsSlot->GetBrickAuxPtr(bid)
            };
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryGetVoxelBrickCached(
            VoxelCollider* collider,
            int3 globalBrickCoord,
            BrickCacheEntry* cache,
            ref VoxelContactQueryCounters counters,
            out VoxelBrickView view)
        {
            counters.BrickLookups++;

            int slot = (globalBrickCoord.x & k_BrickCacheAxisMask)
                | ((globalBrickCoord.y & k_BrickCacheAxisMask) << 2)
                | ((globalBrickCoord.z & k_BrickCacheAxisMask) << 4);

            BrickCacheEntry entry = cache[slot];
            if (entry.Valid && math.all(entry.Coord == globalBrickCoord))
            {
                counters.BrickCacheHits++;
                view = entry.View;
                return entry.Present;
            }

            counters.BrickResolves++;
            bool present = ResolveVoxelBrick(collider, globalBrickCoord, out view);
            cache[slot] = new BrickCacheEntry
            {
                Coord = globalBrickCoord,
                View = view,
                Valid = true,
                Present = present
            };
            return present;
        }

        // -----------------------------------------------------------------------------------
        // Vertex query: source vertex against every active target cell
        // -----------------------------------------------------------------------------------

        // `sourcePointInTarget` is the source voxel center expressed in the TARGET body's grid.
        // Target cells are axis-aligned there, so the closest core point is a clamp.
        private static unsafe void VertexQuery(
            int3 sourceCoord,
            PhysicsInfo sourceInfo,
            VoxelCollider* target,
            float3 sourcePointInTarget,
            in MTransform bFromA,
            float reach,
            float maxDistance,
            bool isSourceFromB,
            BrickCacheEntry* cache,
            ref VoxelContactQueryCounters counters,
            ref UnsafeList<VoxelContact> contacts)
        {
            // The source point must itself be maximal for the budget each target dimension leaves.
            // Bit d of the gate says a target cell of dimension d may pair with this source point.
            int pointGate =
                ((sourceInfo.MaximalFeatureMask(2) & PhysicsInfo.PointMask) != 0 ? 1 : 0) |
                ((sourceInfo.MaximalFeatureMask(1) & PhysicsInfo.PointMask) != 0 ? 2 : 0) |
                ((sourceInfo.MaximalFeatureMask(0) & PhysicsInfo.PointMask) != 0 ? 4 : 0);
            if (pointGate == 0)
            {
                return;
            }

            ComputeTargetRootWindow(
                sourcePointInTarget, sourcePointInTarget, reach, out int3 lower, out int3 upper);

            int recentCount = 0;
            int3 lowerBrick = lower >> Sector.SHIFT_IN_BLOCKS;
            int3 upperBrick = upper >> Sector.SHIFT_IN_BLOCKS;

            for (int brickZ = lowerBrick.z; brickZ <= upperBrick.z; brickZ++)
            {
                for (int brickY = lowerBrick.y; brickY <= upperBrick.y; brickY++)
                {
                    for (int brickX = lowerBrick.x; brickX <= upperBrick.x; brickX++)
                    {
                        int3 brickCoord = new int3(brickX, brickY, brickZ);
                        if (!TryGetVoxelBrickCached(
                                target, brickCoord, cache, ref counters, out VoxelBrickView brick))
                        {
                            continue;
                        }

                        int3 brickOrigin = brickCoord * Sector.SIZE_IN_BLOCKS;
                        int3 localLower = math.max(lower - brickOrigin, int3.zero);
                        int3 localUpper = math.min(upper - brickOrigin, new int3(Sector.BRICK_MASK));
                        if (math.any(localLower > localUpper))
                        {
                            continue;
                        }

                        // The occupancy mask is one ulong per z slice with bit x + 8y, so a whole
                        // voxel row of the window is a contiguous bit field and a whole slice is one
                        // word. Testing those rejects empty space a row or a slice at a time, and
                        // iterating the surviving bits touches only occupied voxels - the sweep is
                        // overwhelmingly empty, so per-voxel probing spends nearly all of its work
                        // on voxels that are not there.
                        int3 localSize = localUpper - localLower + 1;
                        ulong rowSelect = ((1UL << localSize.x) - 1UL) << localLower.x;
                        counters.WindowRoots += localSize.x * localSize.y * localSize.z;

                        for (int z = localLower.z; z <= localUpper.z; z++)
                        {
                            ulong slice = brick.OccupiedMask[z];
                            if (slice == 0UL)
                            {
                                counters.RowsTested += localSize.y;
                                counters.RowsSkipped += localSize.y;
                                continue;
                            }

                            for (int y = localLower.y; y <= localUpper.y; y++)
                            {
                                counters.RowsTested++;
                                ulong row = slice & (rowSelect << (y << 3));
                                if (row == 0UL)
                                {
                                    counters.RowsSkipped++;
                                    continue;
                                }

                                while (row != 0UL)
                                {
                                    int rowBit = math.tzcnt(row);
                                    row &= row - 1UL;

                                    int x = rowBit - (y << 3);
                                    int voxelIndex = (z << 6) | rowBit;
                                    counters.TouchedRoots++;

                                    PhysicsInfo targetInfo = brick.Physics[voxelIndex];

                                    // A vertex source leaves a budget of two, so every active
                                    // surface cell that is maximal at that budget is a candidate.
                                    int allowed = targetInfo.MaximalFeatureMask(
                                        k_MaxPermittedDimensionSum);

                                    // Vertex-vertex pairs are emitted by the A-side pass only.
                                    if (isSourceFromB)
                                    {
                                        allowed &= ~PhysicsInfo.PointMask;
                                    }
                                    if (allowed == 0)
                                    {
                                        continue;
                                    }
                                    counters.ActiveRoots++;

                                    int3 targetRoot = brickOrigin + new int3(x, y, z);
                                    while (allowed != 0)
                                    {
                                        int featureBit = math.tzcnt((uint)allowed);
                                        allowed &= allowed - 1;

                                        int axisMask = PhysicsInfo.AxisMaskFromFeatureBit(featureBit);
                                        int dimension = math.countbits((uint)axisMask);
                                        if ((pointGate & (1 << dimension)) == 0)
                                        {
                                            continue;
                                        }
                                        counters.CellTests++;

                                        GetCellBox(targetRoot, axisMask,
                                            out float3 cellMinimum, out float3 cellMaximum);
                                        float3 witness = math.clamp(
                                            sourcePointInTarget, cellMinimum, cellMaximum);

                                        float3 corePointAinB;
                                        float3 corePointBinB;
                                        if (isSourceFromB)
                                        {
                                            corePointAinB = Mul(bFromA, witness);
                                            // sourcePointInTarget is in A space in this branch. Emission
                                            // requires both witnesses in B space, where the B source point
                                            // is its unchanged voxel center.
                                            corePointBinB = (float3)sourceCoord + 0.5f;
                                        }
                                        else
                                        {
                                            corePointAinB = sourcePointInTarget;
                                            corePointBinB = witness;
                                        }

                                        if (EmitCoreContact(
                                                corePointAinB, corePointBinB,
                                                isSourceFromB ? targetRoot : sourceCoord,
                                                isSourceFromB ? sourceCoord : targetRoot,
                                                (byte)axisMask,
                                                maxDistance, recentCount,
                                                ref counters, ref contacts))
                                        {
                                            recentCount++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------------------
        // Edge-edge query: source segment against every active target segment
        // -----------------------------------------------------------------------------------

        // Runs on the A side only; it covers every (A edge, B edge) pair on its own. Source edges
        // along axis a all share the direction bFromA.Rotation.c_a in B grid space, so the
        // direction is passed in already transformed.
        private static unsafe void EdgeEdgeQuery(
            int3 sourceCoord,
            float3 sourceOriginInB,
            float3 sourceEdgeInB,
            VoxelCollider* voxelB,
            float reach,
            float maxDistance,
            BrickCacheEntry* cache,
            ref VoxelContactQueryCounters counters,
            ref UnsafeList<VoxelContact> contacts)
        {
            float3 sourceMinimum = math.min(sourceOriginInB, sourceOriginInB + sourceEdgeInB);
            float3 sourceMaximum = math.max(sourceOriginInB, sourceOriginInB + sourceEdgeInB);
            ComputeTargetRootWindow(sourceMinimum, sourceMaximum, reach, out int3 lower, out int3 upper);

            int recentCount = 0;
            int3 lowerBrick = lower >> Sector.SHIFT_IN_BLOCKS;
            int3 upperBrick = upper >> Sector.SHIFT_IN_BLOCKS;

            for (int brickZ = lowerBrick.z; brickZ <= upperBrick.z; brickZ++)
            {
                for (int brickY = lowerBrick.y; brickY <= upperBrick.y; brickY++)
                {
                    for (int brickX = lowerBrick.x; brickX <= upperBrick.x; brickX++)
                    {
                        int3 brickCoord = new int3(brickX, brickY, brickZ);
                        if (!TryGetVoxelBrickCached(
                                voxelB, brickCoord, cache, ref counters, out VoxelBrickView brick))
                        {
                            continue;
                        }

                        int3 brickOrigin = brickCoord * Sector.SIZE_IN_BLOCKS;
                        int3 localLower = math.max(lower - brickOrigin, int3.zero);
                        int3 localUpper = math.min(upper - brickOrigin, new int3(Sector.BRICK_MASK));
                        if (math.any(localLower > localUpper))
                        {
                            continue;
                        }

                        // Scanned against the physics KEY mask, not occupancy. This query only
                        // wants targets carrying an active edge, and such a root is a contact source
                        // by definition, so the key mask is a strict superset of them - and a far
                        // sparser one, because most occupied voxels are plain surface or interior
                        // and carry no edge at all.
                        //
                        // Both masks share the BrickBitmask layout: one ulong per z slice with bit
                        // x + 8y, so a whole voxel row of the window is a contiguous bit field and a
                        // whole slice is one word. Testing those rejects empty space a row or a
                        // slice at a time, and iterating the surviving bits reads only candidates.
                        int3 localSize = localUpper - localLower + 1;
                        ulong rowSelect = ((1UL << localSize.x) - 1UL) << localLower.x;
                        counters.WindowRoots += localSize.x * localSize.y * localSize.z;

                        for (int z = localLower.z; z <= localUpper.z; z++)
                        {
                            ulong slice = brick.PhysicsKeyMask[z];
                            if (slice == 0UL)
                            {
                                counters.RowsTested += localSize.y;
                                counters.RowsSkipped += localSize.y;
                                continue;
                            }

                            for (int y = localLower.y; y <= localUpper.y; y++)
                            {
                                counters.RowsTested++;
                                ulong row = slice & (rowSelect << (y << 3));
                                if (row == 0UL)
                                {
                                    counters.RowsSkipped++;
                                    continue;
                                }

                                while (row != 0UL)
                                {
                                    int rowBit = math.tzcnt(row);
                                    row &= row - 1UL;

                                    int x = rowBit - (y << 3);
                                    int voxelIndex = (z << 6) | rowBit;
                                    counters.TouchedRoots++;

                                    PhysicsInfo targetInfo = brick.Physics[voxelIndex];

                                    // An edge source leaves a budget of one. A segment's only
                                    // supersets are squares and cubes, both outside that budget,
                                    // so every active segment here is maximal.
                                    int allowed = targetInfo.data & PhysicsInfo.EdgeMask;
                                    if (allowed == 0)
                                    {
                                        continue;
                                    }
                                    counters.ActiveRoots++;

                                    int3 targetRoot = brickOrigin + new int3(x, y, z);
                                    float3 targetOrigin = (float3)targetRoot + 0.5f;

                                    while (allowed != 0)
                                    {
                                        int featureBit = math.tzcnt((uint)allowed);
                                        allowed &= allowed - 1;
                                        counters.CellTests++;

                                        int axisMask = PhysicsInfo.AxisMaskFromFeatureBit(featureBit);
                                        float3 targetEdge = new float3(
                                            (axisMask & 1) != 0 ? 1.0f : 0.0f,
                                            (axisMask & 2) != 0 ? 1.0f : 0.0f,
                                            (axisMask & 4) != 0 ? 1.0f : 0.0f);

                                        ClosestPointsSegmentSegment(
                                            sourceOriginInB, sourceEdgeInB,
                                            targetOrigin, targetEdge,
                                            out float3 corePointAinB, out float3 corePointBinB);

                                        if (EmitCoreContact(
                                                corePointAinB, corePointBinB,
                                                sourceCoord, targetRoot, (byte)axisMask,
                                                maxDistance, recentCount,
                                                ref counters, ref contacts))
                                        {
                                            recentCount++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------------------
        // Source dispatch
        // -----------------------------------------------------------------------------------

        // Runs both queries for one source voxel. `isSourceFromB` selects which body owns the
        // source; the edge-edge query only runs on the A side, where it covers every edge pair.
        private static unsafe void HandleSourceBlock(
            int3 sourceCoord,
            PhysicsInfo sourceInfo,
            VoxelCollider* voxelA,
            VoxelCollider* voxelB,
            in MTransform bFromA,
            in MTransform aFromB,
            float reach,
            float maxDistance,
            bool isSourceFromB,
            BrickCacheEntry* cache,
            ref VoxelContactCounters counters,
            ref UnsafeList<VoxelContact> contacts)
        {
            float3 sourceCenter = (float3)sourceCoord + 0.5f;

            if (sourceInfo.HasPointFeature)
            {
                VoxelCollider* target = isSourceFromB ? voxelA : voxelB;
                float3 sourcePointInTarget = isSourceFromB
                    ? Mul(aFromB, sourceCenter)
                    : Mul(bFromA, sourceCenter);

                counters.Vertex.Sources++;
                VertexQuery(
                    sourceCoord, sourceInfo, target, sourcePointInTarget,
                    bFromA, reach, maxDistance, isSourceFromB, cache,
                    ref counters.Vertex, ref contacts);
            }

            if (isSourceFromB)
            {
                return;
            }

            int edges = sourceInfo.data & PhysicsInfo.EdgeMask;
            if (edges == 0)
            {
                return;
            }

            float3 originInB = Mul(bFromA, sourceCenter);
            while (edges != 0)
            {
                int featureBit = math.tzcnt((uint)edges);
                edges &= edges - 1;

                // A source segment along axis a maps to the matching rotation column, the same
                // vector for every source voxel of this body pair.
                int axisMask = PhysicsInfo.AxisMaskFromFeatureBit(featureBit);
                float3 edgeInB = (axisMask & 1) != 0 ? bFromA.Rotation.c0
                    : (axisMask & 2) != 0 ? bFromA.Rotation.c1
                    : bFromA.Rotation.c2;

                counters.Edge.Sources++;
                EdgeEdgeQuery(
                    sourceCoord, originInB, edgeInB, voxelB,
                    reach, maxDistance, cache, ref counters.Edge, ref contacts);
            }
        }

        // -----------------------------------------------------------------------------------
        // Pass driver
        // -----------------------------------------------------------------------------------

        // Key-block sourced and brick-culled:
        //
        //   Pass 1 (A-side): every allocated brick of A inside B's reach conservatively marks the
        //   allocated B bricks it may pair with, then each of the brick's physics-KEY blocks runs
        //   its vertex and edge-edge queries against B.
        //   Pass 2 (B-side): the key blocks of every marked B brick run their vertex query against
        //   A, skipping target points because pass 1 already emitted every vertex-vertex pair.
        //
        // A key block is a root carrying an active point or an active edge, which is exactly the
        // source set the two queries need. Probes read the opposite body through its sector map
        // directly, so windows cross sector boundaries without a per-sector-pair loop.
        static unsafe void CollectVoxelContacts(
            VoxelCollider* voxelA,
            VoxelCollider* voxelB,
            in MTransform bFromA,
            float maxDistance,
            ref VoxelContactCounters counters,
            ref UnsafeList<VoxelContact> contacts)
        {
            var sectorsA = voxelA->m_Sectors;

            MTransform aFromB = Inverse(bFromA);

            // Conservative per-axis extent of a rotated A box in B space, for culling only.
            float3x3 rot = bFromA.Rotation;
            float3 rowAbsSum = math.abs(rot.c0) + math.abs(rot.c1) + math.abs(rot.c2);

            float speculative = math.clamp(maxDistance, 0.0f, k_VoxelMaxSpeculativeMargin);

            // Culling is centred on source voxel roots. A rotated source segment can extend by at
            // most sqrt(3) on one B axis, a target cell can extend one cell towards it, and the
            // two radii add one more. The queries compute a tighter window per source feature.
            float cullingWindowHalfWidth = 2.0f + math.sqrt(3.0f) + speculative;
            float reach = 2.0f * k_VoxelCoreRadius + speculative;

            if (!ComputeBodyBoundsInB(voxelB, out float3 boundsCenterB, out float3 boundsHalfB))
            {
                return;
            }

            var overlappedBricksB = new UnsafeHashSet<int3>(64, Allocator.Temp);

            BrickCacheEntry* cacheB = stackalloc BrickCacheEntry[k_BrickCacheSize];
            for (int i = 0; i < k_BrickCacheSize; i++)
            {
                cacheB[i] = default;
            }


            var keysA = sectorsA.GetKeyArray(Allocator.Temp);
            for (int iSectorA = 0; iSectorA < keysA.Length; iSectorA++)
            {
                int3 sectorCoordA = keysA[iSectorA];
                var sectorA = sectorsA[sectorCoordA];
                int3 sectorOriginA = sectorCoordA * Sector.SECTOR_SIZE_IN_BLOCKS;

                if (SectorCannotReachB(
                        sectorOriginA, bFromA, rowAbsSum, cullingWindowHalfWidth,
                        boundsCenterB, boundsHalfB))
                {
                    continue;
                }

                foreach (SectorNonEmptyBrickEnumerator.BrickRef brickRef in sectorA.Ptr->EnumerateNonEmptyBricks())
                {
                    int3 brickOriginBlocks = sectorOriginA
                        + Sector.ToBrickPos((short)brickRef.BrickAbs) * Sector.SIZE_IN_BLOCKS;

                    // Marking is unconditional: a key-less A brick still owns cells that the
                    // B-side pass must reach. Only the key enumeration below is gated on it.
                    bool overlapsB = MarkOverlappedBricksInB(
                        brickOriginBlocks, voxelB, bFromA, rowAbsSum, cullingWindowHalfWidth,
                        ref overlappedBricksB);
                    if (!overlapsB)
                    {
                        continue;
                    }

                    counters.SourceBricks++;

                    foreach (SectorBitmaskSlotIterator<PhysicsInfo> blockIter in
                        sectorA.Ptr->EnumeratePhysicsKeyBlocksInBrick(brickRef.Bid, brickOriginBlocks))
                    {
                        HandleSourceBlock(
                            blockIter.position, blockIter.value, voxelA, voxelB,
                            bFromA, aFromB, reach, maxDistance,
                            isSourceFromB: false, cacheB,
                            ref counters, ref contacts);
                    }
                }
            }
            keysA.Dispose();

            BrickCacheEntry* cacheA = stackalloc BrickCacheEntry[k_BrickCacheSize];
            for (int i = 0; i < k_BrickCacheSize; i++)
            {
                cacheA[i] = default;
            }


            foreach (int3 brickCoordB in overlappedBricksB)
            {
                int3 sectorCoordB = brickCoordB >> Sector.SHIFT_IN_BRICKS;
                int3 brickPosInSector = brickCoordB & Sector.SECTOR_MASK;

                // Present and allocated by construction: only allocated B bricks get marked.
                var sectorB = voxelB->m_Sectors[sectorCoordB];
                short bid = sectorB.Ptr->brickIdx[
                    Sector.ToBrickIdx(brickPosInSector.x, brickPosInSector.y, brickPosInSector.z)];

                int3 brickOriginBlocks = brickCoordB * Sector.SIZE_IN_BLOCKS;
                counters.SourceBricks++;

                foreach (SectorBitmaskSlotIterator<PhysicsInfo> blockIter in
                    sectorB.Ptr->EnumeratePhysicsKeyBlocksInBrick(bid, brickOriginBlocks))
                {
                    HandleSourceBlock(
                        blockIter.position, blockIter.value, voxelA, voxelB,
                        bFromA, aFromB, reach, maxDistance,
                        isSourceFromB: true, cacheA,
                        ref counters, ref contacts);
                }
            }
            overlappedBricksB.Dispose();
        }

        // Whole-body bounds of B in its own grid, at sector granularity, for A-sector culling.
        // Returns false when B holds no sectors at all.
        static unsafe bool ComputeBodyBoundsInB(
            VoxelCollider* voxelB,
            out float3 boundsCenterB,
            out float3 boundsHalfB)
        {
            var keysB = voxelB->m_Sectors.GetKeyArray(Allocator.Temp);
            if (keysB.Length == 0)
            {
                keysB.Dispose();
                boundsCenterB = default;
                boundsHalfB = default;
                return false;
            }
            int3 sectorMinB = keysB[0];
            int3 sectorMaxB = keysB[0];
            for (int i = 1; i < keysB.Length; i++)
            {
                sectorMinB = math.min(sectorMinB, keysB[i]);
                sectorMaxB = math.max(sectorMaxB, keysB[i]);
            }
            keysB.Dispose();
            boundsCenterB = (float3)(sectorMinB + sectorMaxB + 1) * (0.5f * Sector.SECTOR_SIZE_IN_BLOCKS);
            boundsHalfB = (float3)(sectorMaxB + 1 - sectorMinB) * (0.5f * Sector.SECTOR_SIZE_IN_BLOCKS);
            return true;
        }

        // Conservative A-sector-vs-whole-B reachability cull for the contact path.
        static bool SectorCannotReachB(
            int3 sectorOriginA,
            in MTransform bFromA,
            float3 rowAbsSum,
            float cullingWindowHalfWidth,
            float3 boundsCenterB,
            float3 boundsHalfB)
        {
            float3 sectorCenterInB = Mul(bFromA, (float3)sectorOriginA + 0.5f * Sector.SECTOR_SIZE_IN_BLOCKS);
            float3 sectorHalfExtentInB = (0.5f * Sector.SECTOR_SIZE_IN_BLOCKS) * rowAbsSum;
            float3 sectorDelta = math.abs(sectorCenterInB - boundsCenterB);
            return math.any(sectorDelta > sectorHalfExtentInB + boundsHalfB + cullingWindowHalfWidth);
        }

        // Tests one A brick (8^3 blocks, given by its min-corner block coord in A grid) against
        // B's allocated bricks: every allocated B brick whose cells could fall inside the window
        // of some voxel of this brick is added to `overlapped`. Returns whether any was.
        // The bound is the brick's voxel-center cloud in B space (center +- 3.5*|R| per axis)
        // dilated by the per-voxel window; per-axis distance never exceeds center distance in any
        // frame, so this is conservative for both passes' windows.
        static unsafe bool MarkOverlappedBricksInB(
            int3 brickOriginBlocksA,
            VoxelCollider* voxelB,
            in MTransform bFromA,
            float3 rowAbsSum,
            float cullingWindowHalfWidth,
            ref UnsafeHashSet<int3> overlapped)
        {
            GetOverlappingBrickRange(
                brickOriginBlocksA, bFromA, rowAbsSum, cullingWindowHalfWidth,
                out int3 brickLo, out int3 brickHi);

            bool any = false;

            // One-entry sector cache: the (tiny) brick range rarely straddles a sector boundary.
            // TODO: Fold into the future per-thread sector/brick cache shared with voxel queries.
            int3 cachedSectorCoord = default;
            SectorHandle cachedHandle = default;
            bool cacheValid = false;
            bool cachedExists = false;

            for (int gz = brickLo.z; gz <= brickHi.z; gz++)
            {
                for (int gy = brickLo.y; gy <= brickHi.y; gy++)
                {
                    for (int gx = brickLo.x; gx <= brickHi.x; gx++)
                    {
                        int3 brickCoord = new int3(gx, gy, gz);
                        int3 sectorCoord = brickCoord >> Sector.SHIFT_IN_BRICKS;

                        if (!cacheValid || math.any(sectorCoord != cachedSectorCoord))
                        {
                            cachedExists = voxelB->m_Sectors.TryGetValue(sectorCoord, out cachedHandle)
                                && !cachedHandle.IsNull;
                            cachedSectorCoord = sectorCoord;
                            cacheValid = true;
                        }
                        if (!cachedExists)
                        {
                            continue;
                        }

                        int3 p = brickCoord & Sector.SECTOR_MASK;
                        if (cachedHandle.Ptr->brickIdx[Sector.ToBrickIdx(p.x, p.y, p.z)] == Sector.BRICKID_EMPTY)
                        {
                            continue;
                        }

                        overlapped.Add(brickCoord);
                        any = true;
                    }
                }
            }

            return any;
        }

        // Shared by regular contact marking and the explicit post-simulation query API. The caller
        // owns target-brick allocation tests and output policy; this helper only computes the
        // conservative brick range reached by one source brick's voxel-center cloud.
        internal static void GetOverlappingBrickRange(
            int3 brickOriginBlocksA,
            in MTransform bFromA,
            float3 rowAbsSum,
            float windowHalfWidth,
            out int3 brickLo,
            out int3 brickHi)
        {
            float3 centerInB = Mul(
                bFromA,
                (float3)brickOriginBlocksA + 0.5f * Sector.SIZE_IN_BLOCKS);
            float3 halfExtent =
                (0.5f * Sector.SIZE_IN_BLOCKS - 0.5f) * rowAbsSum + windowHalfWidth;

            // B cells whose centers can lie inside the dilated cloud, then the bricks holding them
            // (arithmetic shifts floor-divide correctly for negative coordinates).
            int3 cellLo = (int3)math.ceil(centerInB - halfExtent - 0.5f);
            int3 cellHi = (int3)math.floor(centerInB + halfExtent - 0.5f);
            brickLo = cellLo >> Sector.SHIFT_IN_BLOCKS;
            brickHi = cellHi >> Sector.SHIFT_IN_BLOCKS;
        }

        // -----------------------------------------------------------------------------------
        // Output
        // -----------------------------------------------------------------------------------

        // Writes one single-point manifold and one contact event per raw contact.
        static unsafe void WriteVoxelManifolds(
            in UnsafeList<VoxelContact> contacts,
            Context context,
            in MTransform worldFromB,
            Material materialA,
            Material materialB,
            bool flipped)
        {
            for (int i = 0; i < contacts.Length; i++)
            {
                VoxelContact contact = contacts[i];

                context.VoxelContactWriter->Write(new VoxelContactEventData
                {
                    BodyIndices = context.BodyIndices,
                    VoxelCoordsInA = flipped ? contact.VoxelB : contact.VoxelA,
                    VoxelCoordsInB = flipped ? contact.VoxelA : contact.VoxelB,
                    Normal = (flipped ? -1.0f : 1.0f) * math.mul(worldFromB.Rotation, contact.NormalInB),
                    Distance = contact.Distance,
                    DebugFlags = 0,
                    isPhysicsContact = true,

                    _debug_constraintRecord = contact.TargetAxisMask
                });

                var manifold = new ConvexConvexManifoldQueries.Manifold
                {
                    Normal = math.mul(worldFromB.Rotation, contact.NormalInB)
                };
                manifold[manifold.NumContacts++] = new ContactPoint
                {
                    Position = Mul(worldFromB, contact.PosInB),
                    Distance = contact.Distance
                };

#if SHOW_DEBUG
                Debug.DrawRay(Mul(worldFromB, contact.PosInB), manifold.Normal * 0.5f,
                    contact.Distance > 0 ? Color.cyan : Color.red, 0.0f, false);
#endif

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
            }
        }
    }
}
