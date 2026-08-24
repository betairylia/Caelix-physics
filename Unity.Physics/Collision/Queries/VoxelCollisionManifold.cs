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
        // Collision model: occupied voxel centers form a finite cubical complex. Adjacent centers
        // form segments, four occupied centers form a square, and eight form a cube. Sweeping a
        // sphere of radius 0.5 over that complex gives the collision surface:
        //   * isolated voxels are spheres and one-voxel poles are capsules;
        //   * squares and cubes create exact flat patches;
        //   * an L shape has two connected segments but no fabricated diagonal square;
        //   * every feature is finite, so ownership changes clamp to a shared edge or point instead
        //     of creating or deleting an infinite plane at a footprint threshold.
        //
        // PhysicsInfo stores one bit per cell of that complex ROOTED at the voxel and COLLISION-
        // ACTIVE: bits 0-6 keep the positive octet order (+X, +Y, +Z, +XY, +XZ, +YZ, +XYZ) and bit 7
        // marks the bare point. A cell is active when it owns an outward direction that no cell
        // containing it owns, so flat subdivision edges and vertices are gone while rims, creases,
        // corners, wire ends and sheet faces remain. Bit 6 (the cube) is a VOLUME cell and never a
        // surface feature; a voxel that roots only the cube is deep inside solid and is skipped
        // (PhysicsInfo.IsInterior). The narrowphase finds closest points between these finite convex
        // cores and lets the existing convex-distance query add the two 0.5 radii. This is an
        // analytic local distance field, not a sampled SDF.
        //
        // Only four unordered core-feature pairs are dispatched: vertex-vertex, vertex-edge,
        // vertex-face and edge-edge. Face-face and edge-face are omitted because for two overlapping
        // finite patches the corners of the overlap are always a vertex of one patch inside the
        // other, or a crossing of two boundary edges. That restriction, not any direction test, is
        // what keeps a flat resting patch from producing one contact per tile pair.
        //
        // Active cells rooted at one voxel overlap on purpose -- a solid box corner roots a point
        // lying on three segments lying on three squares -- so the pair loop then keeps only the
        // MAXIMAL permitted pairs of the two roots. Containment between two cells of the same root
        // means the smaller one's constraint is already implied, and the containing cell is built
        // from the same byte in the same loop, so the duplicate is dropped with no risk of a hole.
        // That takes a vertex source against a solid box corner from seven pairs down to three.
        // It does NOT touch duplicates across different roots, which are the seam's problem.
        //
        // NOT PRESENT YET, deliberately (v1 of the active-feature scheme):
        //   * no direction / normal-cone gate, so a feature buried under a coplanar neighbor can
        //     still report a tilted contact;
        //   * no canonical seam ownership (k_EnableSeamOwnership is off), so a witness on a shared
        //     boundary is reported once per cell that touches it.
        // Both omissions only ADD contacts. Generation stays conservative: the union of the emitted
        // constraints is still exactly the non-penetration condition, and no configuration loses a
        // contact it would otherwise have. Removing the duplicates is the merging stage's job.
        //
        // This file intentionally contains GENERATION ONLY: every surviving (A cell, B cell) probe
        // emits one raw contact, written as its own single-point manifold plus one contact event.
        // There is no patch merging or reduction. The previous merging / joint extraction stages
        // (per-(A voxel, sign class) keep-deepest buckets, fusion of opposing contacts into
        // bilateral equality points, per-axis manifold merging, rim/extreme reduction) were removed
        // 2026-07 to make room for a new patch-based merging algorithm; see git history.
        //
        // Containment and deep overlap are NOT covered. Cube cells are excluded from the pair set,
        // so a body buried more than a voxel inside another finds no nearby surface feature and
        // generates nothing. That needs the separate volume path.
        //
        // Conventions (matching the rest of the narrowphase):
        //   * Manifold normals point from B towards A; positive contact distance is separation.
        //   * Positions are in world space, on body B's surface.
        // Uniform body scale is not supported here (assumed 1), same as the previous prototype.
        // ---------------------------------------------------------------------------------------

        // Cap on the speculative margin used to size the candidate-cell window. The contact
        // distance gate still uses the full maxDistance; this only bounds the search volume.
        const float k_VoxelMaxSpeculativeMargin = 0.5f;

        // Current collider model is intentionally fixed to the fully rounded case.
        const float k_VoxelCoreRadius = 0.5f;

        // A root can carry every surface cell at once: the minimum corner voxel of a solid box roots
        // three boundary faces, three convex edges and the corner point. The cube bit is excluded
        // from the surface path, so seven is the bound and four (a square) is the worst vertex count
        // per feature. Sizing by the worst feature rather than the worst reachable combination means
        // a byte from an older layout cannot overflow the caller's stack buffer.
        const int k_MaxFeaturesPerVoxel = PhysicsInfo.SurfaceFeatureBitCount;
        const int k_MaxVerticesPerVoxel = 4 * k_MaxFeaturesPerVoxel;

        // Seam ownership: a witness on a cell's positive boundary belongs to the cell rooted one
        // voxel further along. Disabled for v1 of the active-feature scheme, so a shared boundary is
        // reported once per touching cell. Turning it back on needs its lookup rules re-checked
        // against active features first -- see OwnsCorePoint.
        const bool k_EnableSeamOwnership = false;

        // One raw generated contact, in B grid space.
        struct VoxelContact
        {
            public float Distance;    // separation along the normal (negative = penetration)
            public float3 NormalInB;  // unit normal (B grid), points from B towards A
            public float3 PosInB;     // contact position on B's surface
            public int3 VoxelA;       // contributing A voxel (A grid)
            public int3 VoxelB;       // contributing B voxel (B grid)
            // public bool Diagonal;     // corner/edge rounding contact (not an axis-face contact)

            // Reserved for the future patch reducer. Cubical-feature contacts currently use bin 0.
            public int normalBin;
        }

        // One finite convex cell of the voxel-center cubical complex. AxisMask uses X=1, Y=2,
        // Z=4. A point has mask 0, a segment one bit, a square two bits, and a cube all three.
        struct CubicCoreFeature
        {
            public int VertexOffset;
            public int VertexCount;
            public byte AxisMask;
            // Popcount of AxisMask, cached at build time because the pair filter reads it on every
            // iteration of the feature loop.
            public byte Dimension;
            // 1 << the PhysicsInfo feature bit this cell came from. Cached for the same reason: the
            // maximality filter tests it against a mask from the opposite root on every iteration.
            public byte FeatureBitMask;
            public float3 RootInLocal;
            public float3 MinInB;
            public float3 MaxInB;
        }

        unsafe struct VoxelBrickView
        {
            public PhysicsInfo* Physics;
            public ulong* OccupiedMask;
            public ulong* PhysicsKeyMask;
        }

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

            // Skip if any of them is marked with a "None" collision response
            if (combinedCollisionResponse == CollisionResponsePolicy.None)
            {
                return;
            }

            // Skip if the bodies have infinite mass and the materials don't want to raise any solver events
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

            // Swap the order if A is larger than B
            // TODO: Make this more precise perhaps
            bool isALargerThanB = (voxelA->m_Sectors.Count > voxelB->m_Sectors.Count);
            if (isALargerThanB)
            {
                _VoxelVoxel(
                    context,
                    voxelB,
                    voxelA,
                    materialB,
                    materialA,
                    worldFromB,
                    worldFromA,
                    maxDistance,
                    !flipped);
                return;
            }

            _VoxelVoxel(
                context,
                voxelA,
                voxelB,
                materialA,
                materialB,
                worldFromA,
                worldFromB,
                maxDistance,
                flipped);
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
            CollectVoxelContacts(voxelA, voxelB, bFromA, maxDistance, ref contacts);
            WriteVoxelManifolds(contacts, context, worldFromB, materialA, materialB, flipped);
            contacts.Dispose();
        }
            

        // Adds one feature rooted at voxelCenter. Its vertices are emitted in B space so all
        // feature-pair distance queries use the identity transform.
        private static unsafe void AddCubicCoreFeature(
            float3 voxelCenter,
            byte axisMask,
            bool transformFromA,
            in MTransform bFromA,
            CubicCoreFeature* features,
            ref int featureCount,
            float3* vertices,
            ref int vertexCount)
        {
            CubicCoreFeature feature = new CubicCoreFeature
            {
                VertexOffset = vertexCount,
                VertexCount = 0,
                AxisMask = axisMask,
                Dimension = (byte)math.countbits((uint)axisMask),
                FeatureBitMask = (byte)(1 << PhysicsInfo.FeatureBitFromAxisMask(axisMask)),
                RootInLocal = voxelCenter,
                MinInB = new float3(float.MaxValue),
                MaxInB = new float3(float.MinValue)
            };

            int maxX = (axisMask & 1) != 0 ? 1 : 0;
            int maxY = (axisMask & 2) != 0 ? 1 : 0;
            int maxZ = (axisMask & 4) != 0 ? 1 : 0;
            for (int z = 0; z <= maxZ; z++)
            {
                for (int y = 0; y <= maxY; y++)
                {
                    for (int x = 0; x <= maxX; x++)
                    {
                        float3 vertex = voxelCenter + new float3(x, y, z);
                        float3 vertexInB = transformFromA ? Mul(bFromA, vertex) : vertex;
                        vertices[vertexCount++] = vertexInB;
                        feature.MinInB = math.min(feature.MinInB, vertexInB);
                        feature.MaxInB = math.max(feature.MaxInB, vertexInB);
                        feature.VertexCount++;
                    }
                }
            }

            features[featureCount++] = feature;
        }

        // Emits the active surface cells rooted at this voxel, one feature per set PhysicsInfo bit.
        // The cube bit is masked off: it is a volume cell, it takes part in no permitted pair, and
        // building it would put an 8-vertex feature into a 4-vertex-per-feature budget. Active cells
        // may overlap geometrically -- a box corner roots a point that lies on three edges that lie
        // on three faces -- which is intended: the lower-dimensional ones exist so a permitted pair
        // can be formed at all.
        private static unsafe int BuildCubicCoreFeatures(
            int3 voxelCoord,
            PhysicsInfo info,
            bool transformFromA,
            in MTransform bFromA,
            CubicCoreFeature* features,
            float3* vertices)
        {
            int featureCount = 0;
            int vertexCount = 0;
            float3 voxelCenter = (float3)voxelCoord + 0.5f;

            uint remaining = (uint)(info.data & PhysicsInfo.SurfaceFeatureMask);
            while (remaining != 0u && featureCount < k_MaxFeaturesPerVoxel)
            {
                int featureBit = math.tzcnt(remaining);
                remaining &= remaining - 1u;

                AddCubicCoreFeature(
                    voxelCenter,
                    (byte)PhysicsInfo.AxisMaskFromFeatureBit(featureBit),
                    transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
            }

            return featureCount;
        }

        // Canonical cells use a half-open interval on every positive core axis, so a point on a
        // cell's positive boundary belongs to the cell rooted one voxel further along those axes.
        // That anchor voxel is unique, which is what removes duplicate seam contacts without
        // changing the geometric distance.
        //
        // The anchor can root nothing -- the far end of a bar, the rim of a plate -- and then nothing
        // else would claim the point, so the anchor is read and this cell keeps the point instead.
        // Without that read every feature end would be a hole in the contact surface. The anchor
        // lookup only runs when a point lands exactly on a positive boundary.
        //
        // OFF in v1 (k_EnableSeamOwnership). Two things must be re-checked before it comes back:
        // active features at one root overlap geometrically, so "the anchor roots a covering cell"
        // is no longer the same question as "the anchor can emit this contact"; and a covering cell
        // that forms no permitted pair with the source would claim a point it cannot report, which
        // turns a duplicate into a hole. CoverMaskForAxes already excludes the cube for that reason.
        private static unsafe bool OwnsCorePoint(
            CubicCoreFeature feature,
            float3 pointInLocal,
            int3 voxelCoord,
            VoxelCollider* collider)
        {
            if (!k_EnableSeamOwnership)
            {
                return true;
            }

            const float endpointTolerance = 1e-4f;
            float3 offset = pointInLocal - feature.RootInLocal;

            int forwardAxes = 0;
            int interiorAxes = 0;
            for (int axis = 0; axis < 3; axis++)
            {
                if ((feature.AxisMask & (1 << axis)) == 0)
                {
                    continue;
                }
                if (offset[axis] >= 1.0f - endpointTolerance)
                {
                    forwardAxes |= 1 << axis;
                }
                else if (offset[axis] > endpointTolerance)
                {
                    interiorAxes |= 1 << axis;
                }
            }

            if (forwardAxes == 0)
            {
                return true;
            }

            int3 anchor = voxelCoord + new int3(
                forwardAxes & 1, (forwardAxes >> 1) & 1, (forwardAxes >> 2) & 1);
            return !AnchorRootsCoveringCell(collider, anchor, interiorAxes);
        }

        // True when the anchor voxel roots a surviving cell that spans every axis the point is
        // interior along, i.e. when that voxel really does own the point. A missing brick answers
        // false so the caller keeps the point: a duplicate contact is recoverable, a hole is not.
        private static unsafe bool AnchorRootsCoveringCell(
            VoxelCollider* collider, int3 voxelCoord, int interiorAxes)
        {
            if (!TryGetVoxelBrick(
                    collider, voxelCoord >> Sector.SHIFT_IN_BLOCKS, out VoxelBrickView brick))
            {
                return false;
            }

            int3 local = voxelCoord & Sector.BRICK_MASK;
            return brick.Physics[Sector.ToBlockIdx(local.x, local.y, local.z)]
                .RootsCellCovering(interiorAxes);
        }

        private static unsafe void GetFeatureBounds(
            CubicCoreFeature* features,
            int featureCount,
            out float3 minimum,
            out float3 maximum)
        {
            minimum = features[0].MinInB;
            maximum = features[0].MaxInB;
            for (int i = 1; i < featureCount; i++)
            {
                minimum = math.min(minimum, features[i].MinInB);
                maximum = math.max(maximum, features[i].MaxInB);
            }
        }

        private static unsafe void GetTransformedFeatureBounds(
            CubicCoreFeature* features,
            int featureCount,
            float3* vertices,
            in MTransform destinationFromB,
            out float3 minimum,
            out float3 maximum)
        {
            minimum = new float3(float.MaxValue);
            maximum = new float3(float.MinValue);
            for (int i = 0; i < featureCount; i++)
            {
                CubicCoreFeature feature = features[i];
                for (int vertexIndex = 0; vertexIndex < feature.VertexCount; vertexIndex++)
                {
                    float3 vertex = Mul(
                        destinationFromB,
                        vertices[feature.VertexOffset + vertexIndex]);
                    minimum = math.min(minimum, vertex);
                    maximum = math.max(maximum, vertex);
                }
            }
        }

        // A target feature can extend one positive cell from its root center. Expand the source
        // core bounds by that cell and by the two rounded radii to get a conservative root range.
        private static void GetCandidateVoxelRange(
            float3 sourceMinimum,
            float3 sourceMaximum,
            float roundedReach,
            out int3 lower,
            out int3 upper)
        {
            float3 targetCenterMinimum = sourceMinimum - roundedReach - 1.0f;
            float3 targetCenterMaximum = sourceMaximum + roundedReach;
            lower = (int3)math.ceil(targetCenterMinimum - 0.5f);
            upper = (int3)math.floor(targetCenterMaximum - 0.5f);
        }

        private static bool BuildRoundedCoreResult(
            float3 corePointAinB,
            float3 corePointBinB,
            out DistanceQueries.Result result)
        {
            float3 delta = corePointAinB - corePointBinB;
            float distanceSq = math.lengthsq(delta);
            if (distanceSq <= 1e-8f)
            {
                result = default;
                return false;
            }

            float coreDistance = math.sqrt(distanceSq);
            float3 normalInB = delta / coreDistance;
            result = new DistanceQueries.Result
            {
                NormalInA = normalInB,
                PositionOnAinA = corePointAinB - normalInB * k_VoxelCoreRadius,
                Distance = coreDistance - 2.0f * k_VoxelCoreRadius
            };
            return true;
        }

        // Point-feature pairs and grid-aligned feature pairs cover the common voxel cases. They
        // need only clamps. Rotated feature-feature pairs use the general convex distance query.
        private static unsafe bool TryFastCoreDistance(
            CubicCoreFeature featureA,
            CubicCoreFeature featureB,
            float3* verticesA,
            float3* verticesB,
            in MTransform bFromA,
            in MTransform aFromB,
            out DistanceQueries.Result result)
        {
            if (featureA.VertexCount == 1)
            {
                float3 corePointAinB = verticesA[featureA.VertexOffset];
                float3 corePointBinB = math.clamp(
                    corePointAinB, featureB.MinInB, featureB.MaxInB);
                return BuildRoundedCoreResult(corePointAinB, corePointBinB, out result);
            }

            if (featureB.VertexCount == 1)
            {
                float3 corePointBinB = verticesB[featureB.VertexOffset];
                float3 pointBinA = Mul(aFromB, corePointBinB);
                float3 featureMaxA = featureA.RootInLocal + new float3(
                    (featureA.AxisMask & 1) != 0 ? 1.0f : 0.0f,
                    (featureA.AxisMask & 2) != 0 ? 1.0f : 0.0f,
                    (featureA.AxisMask & 4) != 0 ? 1.0f : 0.0f);
                float3 corePointAinA = math.clamp(
                    pointBinA, featureA.RootInLocal, featureMaxA);
                return BuildRoundedCoreResult(
                    Mul(bFromA, corePointAinA), corePointBinB, out result);
            }

            bool sameOrientation =
                math.all(math.abs(bFromA.Rotation.c0 - new float3(1, 0, 0)) < 1e-5f) &&
                math.all(math.abs(bFromA.Rotation.c1 - new float3(0, 1, 0)) < 1e-5f) &&
                math.all(math.abs(bFromA.Rotation.c2 - new float3(0, 0, 1)) < 1e-5f);
            if (!sameOrientation)
            {
                result = default;
                return false;
            }

            float3 corePointA = math.max(featureA.MinInB, featureB.MinInB);
            corePointA = math.min(corePointA, featureA.MaxInB);
            float3 corePointB = math.clamp(corePointA, featureB.MinInB, featureB.MaxInB);
            corePointA = math.clamp(corePointB, featureA.MinInB, featureA.MaxInB);
            return BuildRoundedCoreResult(corePointA, corePointB, out result);
        }

        // Voxel-vs-voxel raw contact generation, key-block sourced and brick-culled:
        //
        //   Pass 1 (A-side): every allocated brick of A inside B's reach conservatively marks
        //   the allocated B bricks it may pair with, then each of the brick's physics-KEY
        //   blocks probes the B cells within its window.
        //   Pass 2 (B-side): the key blocks of every marked B brick probe A's cells the same
        //   way, skipping A-key targets — those pairs were already emitted by pass 1.
        //
        // A key block is a root carrying an active point or an active edge. That is exactly the
        // source set the permitted pairs need: every one of them has a vertex or an edge on at least
        // one side, so a face-only root is always a target and never a source. The two passes
        // together cover (A-key, any B) and (non-A-key A, B-key); the only pair they cannot reach is
        // face-face, which is not dispatched anyway. Probes read the opposite body through its
        // sector map directly, so windows cross sector boundaries without a per-sector-pair loop.
        static unsafe void CollectVoxelContacts(
            VoxelCollider* voxelA,
            VoxelCollider* voxelB,
            in MTransform bFromA,
            float maxDistance,
            ref UnsafeList<VoxelContact> contacts)
        {
            var sectorsA = voxelA->m_Sectors;

            MTransform aFromB = Inverse(bFromA);

            // Conservative per-axis extent of a rotated A box in B space (for culling only).
            float3x3 rot = bFromA.Rotation;
            float3 rowAbsSum = math.abs(rot.c0) + math.abs(rot.c1) + math.abs(rot.c2);

            float speculative = math.clamp(maxDistance, 0.0f, k_VoxelMaxSpeculativeMargin);

            // Brick and sector culling is centered on source voxel roots. A rotated positive
            // feature can extend by at most sqrt(3) on one B axis. A target feature can extend one
            // cell toward it, and the two radii add one more cell. HandleBlock computes a tighter
            // range from each source feature set after this conservative cull.
            float cullingWindowHalfWidth = 2.0f + math.sqrt(3.0f) + speculative;
            float roundedReach = 2.0f * k_VoxelCoreRadius + speculative;

            // Whole-body bounds of B in its own grid (sector granularity), for A-sector culling.
            if (!ComputeBodyBoundsInB(voxelB, out float3 boundsCenterB, out float3 boundsHalfB))
            {
                return;
            }

            // Allocated B bricks (global brick coords) some A brick may pair with; the B-side
            // pass sources its key blocks from exactly these.
            var overlappedBricksB = new UnsafeHashSet<int3>(64, Allocator.Temp);

            var keysA = sectorsA.GetKeyArray(Allocator.Temp);
            for (int iSectorA = 0; iSectorA < keysA.Length; iSectorA++)
            {
                int3 sectorCoordA = keysA[iSectorA];
                var sectorA = sectorsA[sectorCoordA];
                int3 sectorOriginA = sectorCoordA * Sector.SECTOR_SIZE_IN_BLOCKS;

                // Cull A sectors that cannot reach B at all.
                if (SectorCannotReachB(sectorOriginA, bFromA, rowAbsSum, cullingWindowHalfWidth, boundsCenterB, boundsHalfB))
                {
                    continue;
                }

                foreach (SectorNonEmptyBrickEnumerator.BrickRef brickRef in sectorA.Ptr->EnumerateNonEmptyBricks())
                {
                    int3 brickOriginBlocks = sectorOriginA
                        + Sector.ToBrickPos((short)brickRef.BrickAbs) * Sector.SIZE_IN_BLOCKS;

                    // Brick-level cull. Marking is unconditional: a key-less A brick still owns
                    // Face blocks that the B-side pass must probe. Only the key enumeration
                    // below is gated on the result.
                    bool overlapsB = MarkOverlappedBricksInB(
                        brickOriginBlocks, voxelB, bFromA, rowAbsSum, cullingWindowHalfWidth,
                        ref overlappedBricksB);
                    if (!overlapsB)
                    {
                        continue;
                    }

                    foreach (SectorBitmaskSlotIterator<PhysicsInfo> blockIter in
                        sectorA.Ptr->EnumeratePhysicsKeyBlocksInBrick(brickRef.Bid, brickOriginBlocks))
                    {
                        HandleBlock(
                            blockIter.position, blockIter.value, voxelA, voxelB,
                            bFromA, aFromB, roundedReach, maxDistance,
                            isSourceFromB: false, ref contacts);
                    }
                }
            }
            keysA.Dispose();

            // B-side pass: key blocks of every marked B brick probe A's cells.
            foreach (int3 brickCoordB in overlappedBricksB)
            {
                int3 sectorCoordB = brickCoordB >> Sector.SHIFT_IN_BRICKS;
                int3 brickPosInSector = brickCoordB & Sector.SECTOR_MASK;

                // Present and allocated by construction: only allocated B bricks get marked.
                var sectorB = voxelB->m_Sectors[sectorCoordB];
                short bid = sectorB.Ptr->brickIdx[
                    Sector.ToBrickIdx(brickPosInSector.x, brickPosInSector.y, brickPosInSector.z)];

                int3 brickOriginBlocks = brickCoordB * Sector.SIZE_IN_BLOCKS;
                foreach (SectorBitmaskSlotIterator<PhysicsInfo> blockIter in
                    sectorB.Ptr->EnumeratePhysicsKeyBlocksInBrick(bid, brickOriginBlocks))
                {
                    HandleBlock(
                        blockIter.position, blockIter.value, voxelA, voxelB,
                        bFromA, aFromB, roundedReach, maxDistance,
                        isSourceFromB: true, ref contacts);
                }
            }
            overlappedBricksB.Dispose();
        }

        // Whole-body bounds of B in its own grid (sector granularity), for A-sector culling.
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

        // Tests one A brick (8³ blocks, given by its min-corner block coord in A grid) against
        // B's allocated bricks: every allocated B brick whose cells could fall inside the probe
        // window of some voxel of this brick is added to `overlapped`. Returns whether any was.
        // The bound is the brick's voxel-center cloud in B space (center ± 3.5·|R| per axis)
        // dilated by the per-voxel window; per-axis distance never exceeds center distance in
        // any frame, so this is conservative for both passes' windows.
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

        // Shared by regular contact marking and the explicit post-simulation query API. The
        // caller owns target-brick allocation tests and output policy; this helper only computes
        // the conservative brick range reached by one source brick's voxel-center cloud.
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

            // B cells whose centers can lie inside the dilated cloud, then the bricks holding
            // them (arithmetic shifts floor-divide correctly for negative coordinates).
            int3 cellLo = (int3)math.ceil(centerInB - halfExtent - 0.5f);
            int3 cellHi = (int3)math.floor(centerInB + halfExtent - 0.5f);
            brickLo = cellLo >> Sector.SHIFT_IN_BLOCKS;
            brickHi = cellHi >> Sector.SHIFT_IN_BLOCKS;
        }

        // Resolves one global brick once. Candidate probes then use direct pointers for all cells
        // in that brick instead of repeating a sector hash lookup and brick-map lookup per cell.
        static unsafe bool TryGetVoxelBrick(
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

        static unsafe void ProbeCandidateRange(
            int3 lower,
            int3 upper,
            VoxelCollider* voxelA,
            VoxelCollider* voxelB,
            int3 sourceCoord,
            PhysicsInfo sourceInfo,
            in MTransform bFromA,
            in MTransform aFromB,
            float maxDistance,
            bool isSourceFromB,
            CubicCoreFeature* sourceFeatures,
            int sourceFeatureCount,
            float3* sourceVertices,
            ref UnsafeList<VoxelContact> contacts)
        {
            VoxelCollider* target = isSourceFromB ? voxelA : voxelB;
            int3 lowerBrick = lower >> Sector.SHIFT_IN_BLOCKS;
            int3 upperBrick = upper >> Sector.SHIFT_IN_BLOCKS;
            for (int brickZ = lowerBrick.z; brickZ <= upperBrick.z; brickZ++)
            {
                for (int brickY = lowerBrick.y; brickY <= upperBrick.y; brickY++)
                {
                    for (int brickX = lowerBrick.x; brickX <= upperBrick.x; brickX++)
                    {
                        int3 brickCoord = new int3(brickX, brickY, brickZ);
                        if (!TryGetVoxelBrick(target, brickCoord, out VoxelBrickView brick))
                        {
                            continue;
                        }

                        int3 brickOrigin = brickCoord * Sector.SIZE_IN_BLOCKS;
                        int3 localLower = math.max(lower - brickOrigin, int3.zero);
                        int3 localUpper = math.min(
                            upper - brickOrigin, new int3(Sector.BRICK_MASK));
                        for (int z = localLower.z; z <= localUpper.z; z++)
                        {
                            for (int y = localLower.y; y <= localUpper.y; y++)
                            {
                                for (int x = localLower.x; x <= localUpper.x; x++)
                                {
                                    int voxelIndex = Sector.ToBlockIdx(x, y, z);
                                    if (!BrickBitmask.GetBit(brick.OccupiedMask, voxelIndex))
                                    {
                                        continue;
                                    }

                                    // Key-key pairs were already emitted by the A-side pass.
                                    if (isSourceFromB &&
                                        BrickBitmask.GetBit(brick.PhysicsKeyMask, voxelIndex))
                                    {
                                        continue;
                                    }

                                    // Occupied but rooting no active surface cell: either deep inside
                                    // solid (only the volume cube bit is set) or fully absorbed by
                                    // its neighbors. Either way a nearer cell carries this center's
                                    // geometry, and the probe range already extends one cell towards
                                    // the negative side to reach it.
                                    PhysicsInfo targetInfo = brick.Physics[voxelIndex];
                                    if (!targetInfo.HasSurfaceFeatures)
                                    {
                                        continue;
                                    }

                                    // No permitted feature pair can exist between these two roots, so
                                    // skip before building the target's features and transforming its
                                    // vertices. Two byte reads decide it.
                                    if (!CanRootsFormPermittedPair(sourceInfo, targetInfo))
                                    {
                                        continue;
                                    }

                                    int3 targetCoord = brickOrigin + new int3(x, y, z);
                                    EmitVoxelPairContact(
                                        isSourceFromB ? targetCoord : sourceCoord,
                                        isSourceFromB ? targetInfo : sourceInfo,
                                        isSourceFromB ? sourceCoord : targetCoord,
                                        isSourceFromB ? sourceInfo : targetInfo,
                                        voxelA, voxelB,
                                        bFromA, aFromB, maxDistance, isSourceFromB,
                                        sourceFeatures, sourceFeatureCount, sourceVertices,
                                        ref contacts);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Probes every candidate cell of the opposite body within the window around one source
        // key block and appends the surviving finite-feature contacts. Pair math always runs in B
        // grid space with fixed A/B roles, so both passes emit identical contacts for identical pairs.
        static unsafe void HandleBlock(
            int3 sourceCoord,
            PhysicsInfo sourceInfo,
            VoxelCollider* voxelA,
            VoxelCollider* voxelB,
            in MTransform bFromA,
            in MTransform aFromB,
            float roundedReach,
            float maxDistance,
            bool isSourceFromB,
            ref UnsafeList<VoxelContact> contacts)
        {
            CubicCoreFeature* sourceFeatures = stackalloc CubicCoreFeature[k_MaxFeaturesPerVoxel];
            float3* sourceVertices = stackalloc float3[k_MaxVerticesPerVoxel];
            int sourceFeatureCount = BuildCubicCoreFeatures(
                sourceCoord, sourceInfo, !isSourceFromB, bFromA,
                sourceFeatures, sourceVertices);
            if (sourceFeatureCount == 0)
            {
                // A key always roots a cell; bail out rather than read empty feature bounds if a
                // stale key bit ever survives a rebuild.
                return;
            }

            if (!isSourceFromB)
            {
                // An A key voxel probes B cells. Its feature bounds already live in B space.
                GetFeatureBounds(
                    sourceFeatures, sourceFeatureCount,
                    out float3 sourceMinimum, out float3 sourceMaximum);
                GetCandidateVoxelRange(
                    sourceMinimum, sourceMaximum, roundedReach,
                    out int3 lo, out int3 hi);

                ProbeCandidateRange(
                    lo, hi, voxelA, voxelB, sourceCoord, sourceInfo,
                    bFromA, aFromB, maxDistance, false,
                    sourceFeatures, sourceFeatureCount, sourceVertices, ref contacts);
            }
            else
            {
                // A B key cell probes A voxels. Transform its complete finite feature bounds to A.
                GetTransformedFeatureBounds(
                    sourceFeatures, sourceFeatureCount, sourceVertices, aFromB,
                    out float3 sourceMinimum, out float3 sourceMaximum);
                GetCandidateVoxelRange(
                    sourceMinimum, sourceMaximum, roundedReach,
                    out int3 lo, out int3 hi);

                ProbeCandidateRange(
                    lo, hi, voxelA, voxelB, sourceCoord, sourceInfo,
                    bFromA, aFromB, maxDistance, true,
                    sourceFeatures, sourceFeatureCount, sourceVertices, ref contacts);
            }
        }

        // The four unordered core-feature pairs the normal contact path dispatches -- vertex-vertex,
        // vertex-edge, vertex-face and edge-edge -- are exactly the pairs whose dimensions sum to at
        // most two. Dimension is the popcount of the axis mask, so a point is 0, a segment 1, a
        // square 2 and a cube 3:
        //
        //   V-V 0  V-E 1  V-F 2  E-E 2   permitted
        //   E-F 3  F-F 4  cube 3+        rejected
        //
        // Face-face and edge-face are omitted on purpose. For two overlapping finite patches the
        // corners of the overlap region are always a vertex of one patch lying inside the other, or
        // a crossing of two boundary edges, so this set still reaches every corner of a resting
        // patch -- while a face pair would fire once per overlapping tile pair.
        const int k_MaxPermittedDimensionSum = 2;

        // Cheapest possible rejection, decided from two PhysicsInfo bytes before a single feature or
        // vertex is built. The lowest dimension a root carries bounds every pair it can take part in,
        // so if the two minima already exceed the budget no feature pair can fit either.
        //
        // This is what keeps a flat rest cheap. A rim source carries only edges (minimum 1) and a
        // flat interior target carries only faces (minimum 2), so 1 + 2 > 2 rejects the whole voxel
        // pair -- no feature build, no vertex transforms, no pair loop. Only a source that carries an
        // actual vertex, i.e. a geometric corner, pays for a flat target.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool CanRootsFormPermittedPair(PhysicsInfo source, PhysicsInfo target)
        {
            return source.MinSurfaceFeatureDimension() + target.MinSurfaceFeatureDimension()
                   <= k_MaxPermittedDimensionSum;
        }

        // Same-root maximality masks for one voxel, packed one byte per dimension budget so the pair
        // loop reads them with a shift and an AND. Budget b sits in byte b, and b is always
        // k_MaxPermittedDimensionSum minus the opposite feature's dimension, i.e. 0, 1 or 2.
        //
        // A cell contained in another cell of the SAME root produces a constraint that the containing
        // cell's constraint already implies, so its pair is pure duplication. Both cells come from the
        // same PhysicsInfo byte, so the dominating pair is always enumerated in the same loop below --
        // dropping the dominated one can never open a hole. See PhysicsInfo.MaximalFeatureMask.
        //
        // The mask only ever contains cells of dimension at most b, so testing a feature against it
        // also decides the permitted-pair rule; no separate dimension-sum test is needed.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static uint PackMaximalFeatureMasks(PhysicsInfo info)
        {
            return info.MaximalFeatureMask(0)
                   | ((uint)info.MaximalFeatureMask(1) << 8)
                   | ((uint)info.MaximalFeatureMask(2) << 16);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static byte MaximalFeaturesForBudget(uint packedMasks, int dimensionBudget)
        {
            return (byte)(packedMasks >> (dimensionBudget << 3));
        }

        // Appends one contact per permitted feature pair inside the speculative distance.
        //
        // The FEATURE, not the voxel, is the unit of geometry: an inside corner roots two segments
        // on one voxel, and keeping only the closest pair would drop one of the two distinct support
        // normals. Pairs that resolve to the same point and normal -- two cells of one root meeting
        // on a shared edge -- are still emitted once.
        static unsafe void EmitVoxelPairContact(
            int3 voxelCoordA,
            PhysicsInfo infoA,
            int3 voxelCoordB,
            PhysicsInfo infoB,
            VoxelCollider* voxelA,
            VoxelCollider* voxelB,
            in MTransform bFromA,
            in MTransform aFromB,
            float maxDistance,
            bool isSourceFromB,
            CubicCoreFeature* sourceFeatures,
            int sourceFeatureCount,
            float3* sourceVertices,
            ref UnsafeList<VoxelContact> contacts)
        {
            CubicCoreFeature* targetFeatures = stackalloc CubicCoreFeature[k_MaxFeaturesPerVoxel];
            float3* targetVertices = stackalloc float3[k_MaxVerticesPerVoxel];
            int targetFeatureCount = BuildCubicCoreFeatures(
                isSourceFromB ? voxelCoordA : voxelCoordB,
                isSourceFromB ? infoA : infoB,
                isSourceFromB,
                bFromA,
                targetFeatures,
                targetVertices);

            CubicCoreFeature* featuresA = isSourceFromB ? targetFeatures : sourceFeatures;
            CubicCoreFeature* featuresB = isSourceFromB ? sourceFeatures : targetFeatures;
            float3* verticesA = isSourceFromB ? targetVertices : sourceVertices;
            float3* verticesB = isSourceFromB ? sourceVertices : targetVertices;
            int featureCountA = isSourceFromB ? targetFeatureCount : sourceFeatureCount;
            int featureCountB = isSourceFromB ? sourceFeatureCount : targetFeatureCount;

            float maxCoreDistance = math.max(0.0f, 2.0f * k_VoxelCoreRadius + maxDistance);
            float maxCoreDistanceSq = maxCoreDistance * maxCoreDistance;

            // Kept pairs are exactly the MAXIMAL permitted pairs. Order two pairs by containment on
            // both sides at once; the permitted set is closed downwards under that order, so every
            // rejected pair is implied by a kept one. Growing one side shrinks the other's budget,
            // which is why the test is two-sided: a pair survives when its A cell is maximal for the
            // budget its B cell leaves AND its B cell is maximal for the budget its A cell leaves.
            //
            // Both masks come from the two PhysicsInfo bytes of this voxel pair, so they are hoisted
            // out of both loops. The masks carry only cells inside the budget, so the byte test is
            // also the permitted-pair test. A flat target carries only squares, so a corner source
            // runs its point against them and skips its own segments and squares outright.
            // featuresB always belongs to body B's voxel, in both pass directions.
            uint maximalA = PackMaximalFeatureMasks(infoA);
            uint maximalB = PackMaximalFeatureMasks(infoB);

            int emitted = 0;
            for (int i = 0; i < featureCountA; i++)
            {
                CubicCoreFeature featureA = featuresA[i];
                byte allowedB = MaximalFeaturesForBudget(
                    maximalB, k_MaxPermittedDimensionSum - featureA.Dimension);
                if (allowedB == 0)
                {
                    continue;
                }

                for (int j = 0; j < featureCountB; j++)
                {
                    CubicCoreFeature featureB = featuresB[j];
                    if ((allowedB & featureB.FeatureBitMask) == 0)
                    {
                        continue;
                    }

                    byte allowedA = MaximalFeaturesForBudget(
                        maximalA, k_MaxPermittedDimensionSum - featureB.Dimension);
                    if ((allowedA & featureA.FeatureBitMask) == 0)
                    {
                        continue;
                    }

                    float3 aabbGap = math.max(
                        math.max(featureA.MinInB - featureB.MaxInB,
                                 featureB.MinInB - featureA.MaxInB),
                        float3.zero);
                    if (math.lengthsq(aabbGap) >= maxCoreDistanceSq)
                    {
                        continue;
                    }

                    DistanceQueries.Result candidate;
                    if (!TryFastCoreDistance(
                            featureA, featureB, verticesA, verticesB,
                            bFromA, aFromB, out candidate))
                    {
                        // TODO: Get rid of general GJK.
                        // Do specific queries dispatched by feature type for better performance.
                        candidate = DistanceQueries.ConvexConvex(
                            verticesA + featureA.VertexOffset,
                            featureA.VertexCount,
                            k_VoxelCoreRadius,
                            verticesB + featureB.VertexOffset,
                            featureB.VertexCount,
                            k_VoxelCoreRadius,
                            MTransform.Identity);
                    }

                    if (candidate.Distance >= maxDistance || !math.isfinite(candidate.Distance) ||
                        !math.all(math.isfinite(candidate.NormalInA)) ||
                        !math.all(math.isfinite(candidate.PositionOnBinA)))
                    {
                        continue;
                    }

                    float3 corePointAinB = candidate.PositionOnAinA +
                                           candidate.NormalInA * k_VoxelCoreRadius;
                    float3 corePointBinB = candidate.PositionOnBinA -
                                           candidate.NormalInA * k_VoxelCoreRadius;
                    if (!OwnsCorePoint(featureA, Mul(aFromB, corePointAinB), voxelCoordA, voxelA) ||
                        !OwnsCorePoint(featureB, corePointBinB, voxelCoordB, voxelB))
                    {
                        continue;
                    }

                    var contact = new VoxelContact
                    {
                        Distance = candidate.Distance,
                        NormalInB = candidate.NormalInA,
                        PosInB = candidate.PositionOnBinA,
                        VoxelA = voxelCoordA,
                        VoxelB = voxelCoordB,
                        normalBin = 0
                    };

                    if (IsDuplicateOfRecent(contacts, emitted, contact))
                    {
                        continue;
                    }

                    contacts.Add(contact);
                    emitted++;
                }
            }
        }

        // Two cells of the same root can meet on a shared edge and report the identical closest
        // point. Only the contacts this voxel pair just appended can collide that way, so the scan
        // stays at the tail of the list.
        static bool IsDuplicateOfRecent(
            in UnsafeList<VoxelContact> contacts, int recentCount, in VoxelContact candidate)
        {
            const float duplicateTolerance = 1e-4f;
            for (int i = contacts.Length - recentCount; i < contacts.Length; i++)
            {
                VoxelContact existing = contacts[i];
                if (math.all(math.abs(existing.PosInB - candidate.PosInB) < duplicateTolerance) &&
                    math.all(math.abs(existing.NormalInB - candidate.NormalInB) < duplicateTolerance))
                {
                    return true;
                }
            }
            return false;
        }

        // Writes one single-point manifold and one contact event per raw contact. No merging,
        // no reduction: this is the raw generation output the future merging algorithm builds on.

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

                byte debugFlags = 0;
                // if (contact.Diagonal)
                // {
                //     debugFlags |= VoxelContactEventData.FlagDiagonal;
                // }

                context.VoxelContactWriter->Write(new VoxelContactEventData
                {
                    BodyIndices = context.BodyIndices,
                    VoxelCoordsInA = flipped ? contact.VoxelB : contact.VoxelA,
                    VoxelCoordsInB = flipped ? contact.VoxelA : contact.VoxelB,
                    Normal = (flipped ? -1.0f : 1.0f) * math.mul(worldFromB.Rotation, contact.NormalInB),
                    Distance = contact.Distance,
                    DebugFlags = debugFlags,
                    isPhysicsContact = true,

                    _debug_constraintRecord = contact.normalBin
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
                Debug.DrawRay(Mul(worldFromB, contact.PosInB), manifold.Normal * 0.5f,// * (-contact.Distance) * 100.0f,
                    contact.Distance > 0 ? Color.cyan : Color.red, 0.0f, false);
#endif

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
            }
        }
    }
}
