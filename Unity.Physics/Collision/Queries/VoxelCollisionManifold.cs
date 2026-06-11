// #define SHOW_DEBUG

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
        // Contacts are generated per (A voxel, B grid face direction) and snapped to the voxel
        // lattice so they can be merged aggressively:
        //   * Every A surface voxel probes nearby B voxels. For each overlapping solid B voxel the
        //     contact axis is the B grid axis with the largest separation among the exposed faces
        //     of the B voxel (SAT-style), so normals are always B grid axis aligned.
        //   * All contributions for the same (A voxel, face) collapse into one bucket that keeps
        //     the deepest separation. This is the "floored to voxel center" merge step.
        //   * If the same A voxel holds buckets for BOTH faces of an axis and both gaps are within
        //     k_VoxelEqualityMaxGap (a snug fit, e.g. a peg in a same-width hole), the pair is
        //     fused into a single bilateral (equality) contact at the voxel center. The solver
        //     treats it as a joint along that axis (impulse may be negative), which is what makes
        //     emergent voxel machinery rigid instead of jittering between two opposing contacts.
        //   * Remaining unilateral buckets expand to the 4 lattice corners of the touched face
        //     (a single point cannot provide rotational support); corners shared between
        //     neighboring voxels are deduplicated, and each face direction is written as one
        //     shared manifold (reduced to the support-polygon extremes when above the manifold
        //     point limit).
        //
        // Conventions (matching the rest of the narrowphase):
        //   * Manifold normals point from B towards A; positive contact distance is separation.
        //   * Positions are in world space, on body B's surface.
        //   * PhysicsInfo exposure bits follow NeighborhoodSettings face order:
        //     bit = axis * 2 + (0 for +axis, 1 for -axis); a set bit means the face is exposed.
        //
        // Uniform body scale is not supported here (assumed 1), same as the previous prototype.
        // ---------------------------------------------------------------------------------------

        /// <summary>
        /// Maximum gap (in voxel units, per side) for two exactly opposing contacts on the same A
        /// voxel to be fused into one bilateral (equality) constraint. Voxel geometry quantizes
        /// intentional clearances to whole voxels, so anything below this is a snug fit; transient
        /// solver penetration/separation stays well below it.
        /// </summary>
        internal const float k_VoxelEqualityMaxGap = 0.04f;

        // Shrink applied to the lateral overlap test so exactly-touching side faces (offset is a
        // whole voxel) do not create phantom side contacts from float noise.
        const float k_VoxelLateralEpsilon = 1e-5f;

        // Cap on the speculative margin used to size the candidate-cell window. The contact
        // distance gate still uses the full maxDistance; this only bounds the search volume.
        const float k_VoxelMaxSpeculativeMargin = 0.5f;

        // Accumulated data of one (A voxel, B face direction) contact bucket, in B grid space.
        struct VoxelContactBucket
        {
            public float Distance;   // most penetrating separation among all contributing B voxels
            public float3 PlanePos;  // A voxel center projected onto the supporting B face plane
            public float3 Center;    // A voxel center
        }

        struct VoxelCornerPoint
        {
            public float Distance;
            public float3 Pos;       // B grid space
        }

        struct VoxelEqualityPoint
        {
            public int Axis;
            public float Distance;   // (d(+face) - d(-face)) / 2: drives the voxel to the slack center
            public float3 Pos;       // B grid space
        }

        // Packs a lattice coordinate plus face index (0..5) into a single key.
        // Coordinates are biased into a 20 bit unsigned range (supports +-524287 blocks).
        // Opposing faces of the same coordinate differ only in bit 0 (axis*2 / axis*2+1).
        static ulong PackVoxelFaceKey(int3 coord, int face)
        {
            const int bias = 1 << 19;
            return (ulong)(uint)face
                | ((ulong)((uint)(coord.x + bias) & 0xFFFFF) << 3)
                | ((ulong)((uint)(coord.y + bias) & 0xFFFFF) << 23)
                | ((ulong)((uint)(coord.z + bias) & 0xFFFFF) << 43);
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

            CollisionResponsePolicy combinedCollisionResponse = Material.GetCombinedCollisionResponse(materialA, materialB);

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

            var sectorsA = voxelA->m_Sectors;
            var sectorsB = voxelB->m_Sectors;

            if (!sectorsA.IsCreated || !sectorsB.IsCreated || sectorsA.IsEmpty || sectorsB.IsEmpty)
            {
                return;
            }

            MTransform bFromA = Mul(Inverse(worldFromB), worldFromA);

            // Per-axis support radius of a (rotated) unit A voxel cube measured along B's grid
            // axes: half the absolute row sums of the rotation. Identity rotation gives 0.5.
            float3x3 rot = bFromA.Rotation;
            float3 rowAbsSum = math.abs(rot.c0) + math.abs(rot.c1) + math.abs(rot.c2);
            float3 supportA = 0.5f * rowAbsSum;

            float speculative = math.clamp(maxDistance, 0.0f, k_VoxelMaxSpeculativeMargin);
            float3 windowHalfWidth = 0.5f + supportA + speculative;

            var buckets = new UnsafeHashMap<ulong, VoxelContactBucket>(512, Allocator.Temp);

            var keysA = sectorsA.GetKeyArray(Allocator.Temp);
            var keysB = sectorsB.GetKeyArray(Allocator.Temp);

            for (int iSectorA = 0; iSectorA < keysA.Length; iSectorA++)
            {
                int3 sectorCoordA = keysA[iSectorA];
                var sectorA = sectorsA[sectorCoordA];
                int3 sectorOriginA = sectorCoordA * Sector.SECTOR_SIZE_IN_BLOCKS;

                // Conservative bounds of this A sector in B grid space, for sector pair culling.
                float3 sectorCenterInB = Mul(bFromA, (float3)sectorOriginA + 0.5f * Sector.SECTOR_SIZE_IN_BLOCKS);
                float3 sectorHalfExtentInB = (0.5f * Sector.SECTOR_SIZE_IN_BLOCKS) * rowAbsSum;

                for (int iSectorB = 0; iSectorB < keysB.Length; iSectorB++)
                {
                    int3 sectorCoordB = keysB[iSectorB];
                    int3 sectorOriginB = sectorCoordB * Sector.SECTOR_SIZE_IN_BLOCKS;

                    // Cull sector pairs that cannot possibly touch.
                    float3 sectorDelta = math.abs(
                        sectorCenterInB - ((float3)sectorOriginB + 0.5f * Sector.SECTOR_SIZE_IN_BLOCKS));
                    if (math.any(sectorDelta >
                        sectorHalfExtentInB + 0.5f * Sector.SECTOR_SIZE_IN_BLOCKS + windowHalfWidth))
                    {
                        continue;
                    }

                    var sectorB = sectorsB[sectorCoordB];

                    foreach (Voxelis.BlockIterator blockIter in new Voxelis.SectorNonEmptyBlockEnumerator(*sectorA.Ptr))
                    {
                        int3 posInA = blockIter.position;

                        // Fully interior A voxels (no exposed faces) cannot make surface contact;
                        // skipping them turns O(volume) into O(surface) for large bodies. Note this
                        // relies on the PhysicsInfo slot being up to date (RefreshPhysicsSlot).
                        PhysicsInfo infoA = sectorA.GetSlot<PhysicsInfo>(
                            SectorSlotId.PhysicsInfo, posInA.x, posInA.y, posInA.z);
                        if (infoA.data == 0)
                        {
                            continue;
                        }

                        int3 voxelCoordA = sectorOriginA + posInA;
                        float3 centerInB = Mul(bFromA, (float3)voxelCoordA + 0.5f);
                        float3 centerLocalB = centerInB - (float3)sectorOriginB;

                        // Candidate B cells whose centers lie within the search window, clamped to
                        // this sector (other sectors of B get their own pass of the outer loop).
                        int3 lo = (int3)math.ceil(centerLocalB - 0.5f - windowHalfWidth);
                        int3 hi = (int3)math.floor(centerLocalB - 0.5f + windowHalfWidth);
                        lo = math.max(lo, 0);
                        hi = math.min(hi, Sector.SECTOR_SIZE_IN_BLOCKS - 1);
                        if (math.any(lo > hi))
                        {
                            continue;
                        }

                        for (int bz = lo.z; bz <= hi.z; bz++)
                        {
                            for (int by = lo.y; by <= hi.y; by++)
                            {
                                for (int bx = lo.x; bx <= hi.x; bx++)
                                {
                                    var dstBlock = sectorB.GetBlock(bx, by, bz);
                                    if (dstBlock.isEmpty)
                                    {
                                        continue;
                                    }

                                    PhysicsInfo infoB = sectorB.GetSlot<PhysicsInfo>(
                                        SectorSlotId.PhysicsInfo, bx, by, bz);

                                    float3 delta = centerLocalB - (new float3(bx, by, bz) + 0.5f);

                                    // Choose the exposed B face axis with the largest separation.
                                    // Interior faces (bit not set) can never push, which also means
                                    // deep interior overlap resolves sideways through the nearest
                                    // exposed surface instead of inventing a fake normal.
                                    int bestFace = -1;
                                    float bestDistance = float.MinValue;
                                    for (int axis = 0; axis < 3; axis++)
                                    {
                                        float d = delta[axis];
                                        if (d == 0.0f)
                                        {
                                            continue; // ambiguous side, another axis will handle it
                                        }

                                        int face = axis * 2 + (d > 0.0f ? 0 : 1);
                                        if ((infoB.data & (1 << face)) == 0)
                                        {
                                            continue;
                                        }

                                        int u = axis == 0 ? 1 : 0;
                                        int v = axis == 2 ? 1 : 2;
                                        if (math.abs(delta[u]) >= 0.5f + supportA[u] - k_VoxelLateralEpsilon ||
                                            math.abs(delta[v]) >= 0.5f + supportA[v] - k_VoxelLateralEpsilon)
                                        {
                                            continue; // faces do not overlap laterally
                                        }

                                        float separation = math.abs(d) - (0.5f + supportA[axis]);
                                        if (separation > bestDistance)
                                        {
                                            bestDistance = separation;
                                            bestFace = face;
                                        }
                                    }

                                    bool isPhysicsContact = bestFace >= 0 && bestDistance < maxDistance;
                                    float3 worldNormal = float3.zero;

                                    if (isPhysicsContact)
                                    {
                                        int axis = bestFace >> 1;
                                        float sgn = (bestFace & 1) == 0 ? 1.0f : -1.0f;

                                        float3 normalInB = float3.zero;
                                        normalInB[axis] = sgn;
                                        worldNormal = math.mul(worldFromB.Rotation, normalInB);

                                        // A voxel center projected onto the supporting B face plane.
                                        float3 planePos = centerInB;
                                        planePos[axis] = sectorOriginB[axis] +
                                            (axis == 0 ? bx : (axis == 1 ? by : bz)) + 0.5f + sgn * 0.5f;

                                        ulong key = PackVoxelFaceKey(voxelCoordA, bestFace);
                                        if (buckets.TryGetValue(key, out VoxelContactBucket bucket))
                                        {
                                            if (bestDistance < bucket.Distance)
                                            {
                                                bucket.Distance = bestDistance;
                                                bucket.PlanePos = planePos;
                                                buckets[key] = bucket;
                                            }
                                        }
                                        else
                                        {
                                            buckets.Add(key, new VoxelContactBucket
                                            {
                                                Distance = bestDistance,
                                                PlanePos = planePos,
                                                Center = centerInB
                                            });
                                        }
                                    }

                                    // Per block pair event for gameplay (damage etc.), independent
                                    // of how the physics contacts get merged above.
                                    int3 voxelCoordB = sectorOriginB + new int3(bx, by, bz);
                                    context.VoxelContactWriter->Write(new VoxelContactEventData
                                    {
                                        BodyIndices = context.BodyIndices,
                                        VoxelCoordsInA = flipped ? voxelCoordB : voxelCoordA,
                                        VoxelCoordsInB = flipped ? voxelCoordA : voxelCoordB,
                                        Normal = worldNormal,
                                        isPhysicsContact = isPhysicsContact
                                    });
                                }
                            }
                        }
                    }
                }
            }

            if (!buckets.IsEmpty)
            {
                EmitVoxelContactManifolds(ref buckets, context, worldFromB, materialA, materialB, flipped);
            }

            buckets.Dispose();
            keysA.Dispose();
            keysB.Dispose();
        }

        // Fuses opposing buckets into bilateral constraints, expands the rest to deduplicated face
        // corner points, and writes one shared manifold per face direction / equality axis.
        static unsafe void EmitVoxelContactManifolds(
            ref UnsafeHashMap<ulong, VoxelContactBucket> buckets,
            Context context,
            in MTransform worldFromB,
            Material materialA,
            Material materialB,
            bool flipped)
        {
            var kv = buckets.GetKeyValueArrays(Allocator.Temp);

            // Pass 1: fuse exactly opposing buckets (same A voxel, same axis, both faces, snug fit)
            // into single bilateral points at the voxel center.
            var consumedKeys = new UnsafeHashSet<ulong>(16, Allocator.Temp);
            var equalityPoints = new UnsafeList<VoxelEqualityPoint>(16, Allocator.Temp);

            for (int i = 0; i < kv.Length; i++)
            {
                ulong key = kv.Keys[i];
                if ((key & 1) != 0)
                {
                    continue; // visit +faces only; the -face partner is looked up below
                }

                ulong oppositeKey = key | 1;
                if (!buckets.TryGetValue(oppositeKey, out VoxelContactBucket negBucket))
                {
                    continue;
                }

                VoxelContactBucket posBucket = kv.Values[i];
                if (math.max(posBucket.Distance, negBucket.Distance) > k_VoxelEqualityMaxGap)
                {
                    continue; // loose fit: keep both unilateral contacts so the voxel may rattle
                }

                consumedKeys.Add(key);
                consumedKeys.Add(oppositeKey);
                equalityPoints.Add(new VoxelEqualityPoint
                {
                    Axis = (int)(key & 7) >> 1,
                    // Drives the voxel towards the center of the remaining slack; both opposing
                    // sides agree on this target so they cannot fight each other.
                    Distance = 0.5f * (posBucket.Distance - negBucket.Distance),
                    Pos = posBucket.Center
                });
            }

            // Pass 2: expand the remaining unilateral buckets into face corner points on the
            // half-voxel lattice. A single center point cannot resist tipping, so each face
            // contributes its 4 corners; corners shared between neighboring voxels merge (keeping
            // the deepest distance), so an NxM patch costs (N+1)x(M+1) points before reduction.
            var corners = new UnsafeHashMap<ulong, VoxelCornerPoint>(kv.Length * 4, Allocator.Temp);
            for (int i = 0; i < kv.Length; i++)
            {
                ulong key = kv.Keys[i];
                if (consumedKeys.Contains(key))
                {
                    continue;
                }

                VoxelContactBucket bucket = kv.Values[i];
                int face = (int)(key & 7);
                int axis = face >> 1;
                int u = axis == 0 ? 1 : 0;
                int v = axis == 2 ? 1 : 2;

                for (int cu = -1; cu <= 1; cu += 2)
                {
                    for (int cv = -1; cv <= 1; cv += 2)
                    {
                        float3 p = bucket.PlanePos;
                        p[u] += 0.5f * cu;
                        p[v] += 0.5f * cv;

                        ulong cornerKey = PackVoxelFaceKey((int3)math.round(p * 2.0f), face);
                        if (corners.TryGetValue(cornerKey, out VoxelCornerPoint corner))
                        {
                            if (bucket.Distance < corner.Distance)
                            {
                                corner.Distance = bucket.Distance;
                                corners[cornerKey] = corner;
                            }
                        }
                        else
                        {
                            corners.Add(cornerKey, new VoxelCornerPoint
                            {
                                Distance = bucket.Distance,
                                Pos = p
                            });
                        }
                    }
                }
            }

            // Pass 3: one bilateral manifold per axis (chunked if above the manifold point limit;
            // equality points are load bearing joints, so they are never reduced away).
            for (int axis = 0; axis < 3; axis++)
            {
                float3 normalInB = float3.zero;
                normalInB[axis] = 1.0f;

                var manifold = new ConvexConvexManifoldQueries.Manifold
                {
                    Normal = math.mul(worldFromB.Rotation, normalInB)
                };

                for (int i = 0; i < equalityPoints.Length; i++)
                {
                    if (equalityPoints[i].Axis != axis)
                    {
                        continue;
                    }

                    if (manifold.NumContacts == ConvexConvexManifoldQueries.Manifold.k_MaxNumContacts)
                    {
                        WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB,
                            flipped, JacobianFlags.IsBilateral);
                        manifold.NumContacts = 0;
                    }

                    manifold[manifold.NumContacts++] = new ContactPoint
                    {
                        Position = Mul(worldFromB, equalityPoints[i].Pos),
                        Distance = equalityPoints[i].Distance
                    };

#if SHOW_DEBUG
                    Debug.DrawRay(Mul(worldFromB, equalityPoints[i].Pos), manifold.Normal, Color.cyan, 0.0f, false);
#endif
                }

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB,
                    flipped, JacobianFlags.IsBilateral);
            }

            // Pass 4: one unilateral manifold per face direction, reduced to the support polygon
            // extremes plus the deepest points when above the manifold point limit.
            var cornerKv = corners.GetKeyValueArrays(Allocator.Temp);
            var facePoints = new UnsafeList<VoxelCornerPoint>(math.max(1, cornerKv.Length), Allocator.Temp);

            for (int face = 0; face < 6; face++)
            {
                facePoints.Clear();
                for (int i = 0; i < cornerKv.Length; i++)
                {
                    if ((int)(cornerKv.Keys[i] & 7) == face)
                    {
                        facePoints.Add(cornerKv.Values[i]);
                    }
                }

                if (facePoints.Length == 0)
                {
                    continue;
                }

                int axis = face >> 1;
                float3 normalInB = float3.zero;
                normalInB[axis] = (face & 1) == 0 ? 1.0f : -1.0f;

                var manifold = new ConvexConvexManifoldQueries.Manifold
                {
                    Normal = math.mul(worldFromB.Rotation, normalInB)
                };

                if (facePoints.Length > ConvexConvexManifoldQueries.Manifold.k_MaxNumContacts)
                {
                    ReduceFacePoints(ref facePoints, axis, ConvexConvexManifoldQueries.Manifold.k_MaxNumContacts);
                }

                for (int i = 0; i < facePoints.Length; i++)
                {
                    manifold[manifold.NumContacts++] = new ContactPoint
                    {
                        Position = Mul(worldFromB, facePoints[i].Pos),
                        Distance = facePoints[i].Distance
                    };

#if SHOW_DEBUG
                    Debug.DrawRay(Mul(worldFromB, facePoints[i].Pos), manifold.Normal * 0.5f, Color.red, 0.0f, false);
#endif
                }

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
            }

            facePoints.Dispose();
            cornerKv.Dispose();
            corners.Dispose();
            equalityPoints.Dispose();
            consumedKeys.Dispose();
            kv.Dispose();
        }

        // In-place selection of up to maxPoints from a face point set: the 8 lateral extreme
        // points (the support polygon rim, which gives rocking stability) plus the deepest of the
        // rest. Order within the list is not meaningful.
        static void ReduceFacePoints(ref UnsafeList<VoxelCornerPoint> points, int axis, int maxPoints)
        {
            int u = axis == 0 ? 1 : 0;
            int v = axis == 2 ? 1 : 2;

            var keep = new NativeArray<bool>(points.Length, Allocator.Temp, NativeArrayOptions.ClearMemory);
            int kept = 0;

            // 8 extreme directions in the face plane: +-u, +-v and the four diagonals
            // ((du, dv) over {-1,0,1}^2 minus the center).
            for (int dir = 0; dir < 9; dir++)
            {
                if (dir == 4)
                {
                    continue;
                }

                float du = dir / 3 - 1;
                float dv = dir % 3 - 1;

                int best = -1;
                float bestScore = float.MinValue;
                for (int i = 0; i < points.Length; i++)
                {
                    float score = points[i].Pos[u] * du + points[i].Pos[v] * dv;
                    if (score > bestScore)
                    {
                        bestScore = score;
                        best = i;
                    }
                }

                if (best >= 0 && !keep[best])
                {
                    keep[best] = true;
                    kept++;
                }
            }

            // Fill the remaining slots with the deepest unselected points.
            while (kept < maxPoints && kept < points.Length)
            {
                int best = -1;
                float bestDistance = float.MaxValue;
                for (int i = 0; i < points.Length; i++)
                {
                    if (!keep[i] && points[i].Distance < bestDistance)
                    {
                        bestDistance = points[i].Distance;
                        best = i;
                    }
                }

                if (best < 0)
                {
                    break;
                }

                keep[best] = true;
                kept++;
            }

            // Compact kept points to the front and shrink.
            int write = 0;
            for (int i = 0; i < points.Length; i++)
            {
                if (keep[i])
                {
                    points[write++] = points[i];
                }
            }

            points.Resize(write, NativeArrayOptions.UninitializedMemory);
            keep.Dispose();
        }
    }
}
