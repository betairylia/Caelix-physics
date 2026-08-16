#define SHOW_DEBUG

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
        // PhysicsInfo's low seven bits store the occupied positive 2x2x2 octet. Each voxel owns the
        // point/segment/square/cube features rooted at its center. The narrowphase finds closest
        // points between these finite convex cores and lets the existing convex-distance query add
        // the two 0.5 radii. This is an analytic local distance field, not a sampled SDF.
        //
        // This file intentionally contains GENERATION ONLY: every surviving (A voxel, B cell)
        // probe emits one raw contact, written as its own single-point manifold plus one contact
        // event. Canonical half-open cells remove exact seam duplicates. There is no patch merging
        // or reduction. The previous
        // merging / joint extraction stages (per-(A voxel, sign class) keep-deepest buckets,
        // fusion of opposing contacts into bilateral equality points, per-axis manifold merging,
        // rim/extreme reduction) were removed 2026-07 to make room for a new patch-based merging
        // algorithm; see git history for the old implementation.
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

        // Builds the maximal positive cubical cells rooted at one voxel. A point is kept only when
        // no positive feature starts here. End voxels therefore still represent the last sphere of
        // a chain, while squares and cubes replace their contained positive edges.
        private static unsafe int BuildCubicCoreFeatures(
            int3 voxelCoord,
            PhysicsInfo info,
            bool transformFromA,
            in MTransform bFromA,
            CubicCoreFeature* features,
            float3* vertices)
        {
            byte occupied = info.ForwardOccupancy;
            bool edgeX = (occupied & (1 << 0)) != 0;
            bool edgeY = (occupied & (1 << 1)) != 0;
            bool edgeZ = (occupied & (1 << 2)) != 0;
            bool squareXY = edgeX && edgeY && (occupied & (1 << 3)) != 0;
            bool squareXZ = edgeX && edgeZ && (occupied & (1 << 4)) != 0;
            bool squareYZ = edgeY && edgeZ && (occupied & (1 << 5)) != 0;
            bool cube = squareXY && squareXZ && squareYZ && (occupied & (1 << 6)) != 0;

            int featureCount = 0;
            int vertexCount = 0;
            float3 voxelCenter = (float3)voxelCoord + 0.5f;

            if (cube)
            {
                AddCubicCoreFeature(
                    voxelCenter, 7, transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
                return featureCount;
            }

            if (squareXY)
            {
                AddCubicCoreFeature(
                    voxelCenter, 3, transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
            }
            if (squareXZ)
            {
                AddCubicCoreFeature(
                    voxelCenter, 5, transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
            }
            if (squareYZ)
            {
                AddCubicCoreFeature(
                    voxelCenter, 6, transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
            }

            if (edgeX && !squareXY && !squareXZ)
            {
                AddCubicCoreFeature(
                    voxelCenter, 1, transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
            }
            if (edgeY && !squareXY && !squareYZ)
            {
                AddCubicCoreFeature(
                    voxelCenter, 2, transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
            }
            if (edgeZ && !squareXZ && !squareYZ)
            {
                AddCubicCoreFeature(
                    voxelCenter, 4, transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
            }

            if (featureCount == 0)
            {
                AddCubicCoreFeature(
                    voxelCenter, 0, transformFromA, bFromA,
                    features, ref featureCount, vertices, ref vertexCount);
            }

            return featureCount;
        }

        // Canonical cells use a half-open interval on every positive core axis. A neighboring
        // feature owns the positive endpoint. End voxels own that point through their point
        // feature. This removes duplicate seam contacts without changing the geometric distance.
        private static bool OwnsCorePoint(CubicCoreFeature feature, float3 pointInLocal)
        {
            const float endpointTolerance = 1e-4f;
            float3 offset = pointInLocal - feature.RootInLocal;
            for (int axis = 0; axis < 3; axis++)
            {
                if ((feature.AxisMask & (1 << axis)) != 0 && offset[axis] >= 1.0f - endpointTolerance)
                {
                    return false;
                }
            }
            return true;
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
        //   blocks (Corner/Edge) probes the B cells within its window.
        //   Pass 2 (B-side): the key blocks of every marked B brick probe A's cells the same
        //   way, skipping A-key targets — those pairs were already emitted by pass 1.
        //
        // Corner and edge blocks remain the sparse contact sources. Every probed target block can
        // contribute its finite point/segment/square/cube core, so broad flat targets stay flat
        // without making every face voxel a source. The two passes make either body's sparse keys
        // available as sources. Probes read the opposite body through its sector map directly, so
        // windows cross sector boundaries without a per-sector-pair loop.
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
            VoxelCollider* target,
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

                                    int3 targetCoord = brickOrigin + new int3(x, y, z);
                                    PhysicsInfo targetInfo = brick.Physics[voxelIndex];
                                    EmitVoxelPairContact(
                                        isSourceFromB ? targetCoord : sourceCoord,
                                        isSourceFromB ? targetInfo : sourceInfo,
                                        isSourceFromB ? sourceCoord : targetCoord,
                                        isSourceFromB ? sourceInfo : targetInfo,
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
            CubicCoreFeature* sourceFeatures = stackalloc CubicCoreFeature[4];
            float3* sourceVertices = stackalloc float3[24];
            int sourceFeatureCount = BuildCubicCoreFeatures(
                sourceCoord, sourceInfo, !isSourceFromB, bFromA,
                sourceFeatures, sourceVertices);

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
                    lo, hi, voxelB, sourceCoord, sourceInfo,
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
                    lo, hi, voxelA, sourceCoord, sourceInfo,
                    bFromA, aFromB, maxDistance, true,
                    sourceFeatures, sourceFeatureCount, sourceVertices, ref contacts);
            }
        }

        // Finds the closest pair of finite cubical-core features rooted at these voxels and
        // appends one contact for the closest pair inside the speculative distance.
        static unsafe void EmitVoxelPairContact(
            int3 voxelCoordA,
            PhysicsInfo infoA,
            int3 voxelCoordB,
            PhysicsInfo infoB,
            in MTransform bFromA,
            in MTransform aFromB,
            float maxDistance,
            bool isSourceFromB,
            CubicCoreFeature* sourceFeatures,
            int sourceFeatureCount,
            float3* sourceVertices,
            ref UnsafeList<VoxelContact> contacts)
        {
            CubicCoreFeature* targetFeatures = stackalloc CubicCoreFeature[4];
            float3* targetVertices = stackalloc float3[24];
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

            float bestDistance = float.MaxValue;
            float maxCoreDistance = math.max(0.0f, 2.0f * k_VoxelCoreRadius + maxDistance);
            float maxCoreDistanceSq = maxCoreDistance * maxCoreDistance;
            DistanceQueries.Result best = default;
            for (int i = 0; i < featureCountA; i++)
            {
                CubicCoreFeature featureA = featuresA[i];
                for (int j = 0; j < featureCountB; j++)
                {
                    CubicCoreFeature featureB = featuresB[j];
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
                        candidate = DistanceQueries.ConvexConvex(
                            verticesA + featureA.VertexOffset,
                            featureA.VertexCount,
                            k_VoxelCoreRadius,
                            verticesB + featureB.VertexOffset,
                            featureB.VertexCount,
                            k_VoxelCoreRadius,
                            MTransform.Identity);
                    }

                    float3 corePointAinB = candidate.PositionOnAinA +
                                           candidate.NormalInA * k_VoxelCoreRadius;
                    float3 corePointBinB = candidate.PositionOnBinA -
                                           candidate.NormalInA * k_VoxelCoreRadius;
                    bool sourceOwnsPoint = isSourceFromB
                        ? OwnsCorePoint(featureB, corePointBinB)
                        : OwnsCorePoint(featureA, Mul(aFromB, corePointAinB));
                    bool targetOwnsPoint = isSourceFromB
                        ? OwnsCorePoint(featureA, Mul(aFromB, corePointAinB))
                        : OwnsCorePoint(featureB, corePointBinB);
                    if (!sourceOwnsPoint || !targetOwnsPoint)
                    {
                        continue;
                    }

                    if (candidate.Distance < bestDistance)
                    {
                        bestDistance = candidate.Distance;
                        best = candidate;
                    }
                }
            }

            if (bestDistance >= maxDistance || !math.isfinite(bestDistance) ||
                !math.all(math.isfinite(best.NormalInA)) ||
                !math.all(math.isfinite(best.PositionOnBinA)))
            {
                return;
            }

            contacts.Add(new VoxelContact
            {
                Distance = bestDistance,
                NormalInB = best.NormalInA,
                PosInB = best.PositionOnBinA,
                VoxelA = voxelCoordA,
                VoxelB = voxelCoordB,
                normalBin = 0
            });
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
                Debug.DrawRay(Mul(worldFromB, contact.PosInB), manifold.Normal * 0.5f * (-contact.Distance) * 100.0f,
                    contact.Distance > 0 ? Color.cyan : Color.red, 0.0f, false);
#endif

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
            }
        }
    }
}
