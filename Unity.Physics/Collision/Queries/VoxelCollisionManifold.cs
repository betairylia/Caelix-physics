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
        // Collision model: every voxel is a SPHERE of radius 0.5 at the voxel center, with the
        // center-to-center direction masked by the touched voxel's face-exposure bits. This is
        // deliberate ("rounded" voxels):
        //   * Masking interior directions makes flat surfaces act flat (a voxel resting anywhere
        //     on a plane sees a pure face normal and a face-plane distance), while real corners
        //     and edges keep their diagonal normals, so voxel entities behave as if their corners
        //     were rounded - a 1-wide pole spins/rolls freely in a 1-wide hole.
        //   * Spheres are rotation invariant, so rotated bodies never jam in snug spaces and the
        //     contact distance is exact regardless of relative orientation.
        //
        // Contacts are snapped to the voxel lattice and merged aggressively:
        //   * Every (A voxel, masked-direction sign class) pair collapses into one bucket keeping
        //     the deepest separation ("floored to voxel center" merging). The sign class is the
        //     per-axis sign of the masked direction (trinary, 26 classes).
        //   * Buckets of the 6 axis-aligned classes share one manifold per direction (their
        //     normals are exact); diagonal classes (corner/edge rounding) emit single-point
        //     manifolds.
        //   * If one A voxel holds buckets for two exactly opposing classes and both are touching
        //     within a small slop (a snug fit, e.g. a peg in a same-width hole), the pair fuses
        //     into a single bilateral (equality) contact at the voxel center. The solver treats it
        //     as a joint along that direction (impulse may be negative): emergent voxel machinery
        //     becomes rigid instead of jittering between two opposing contacts, while the
        //     unconstrained axes still slide/spin freely.
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

        /// <summary>
        /// Maximum penetration (per side) for opposing contacts to still fuse into an equality
        /// constraint. Squeezes deeper than this (i.e. genuinely overlapping bodies) fall back to
        /// two pushing contacts so the overlap resolves instead of getting glued in place.
        /// </summary>
        internal const float k_VoxelEqualityMaxPenetration = 0.25f;

        // Opposing buckets only fuse when their normals are truly anti-parallel. Axis-aligned
        // classes always are; diagonal (rounded corner) classes only when geometry pinches.
        const float k_VoxelEqualityMinOpposition = -0.95f;

        // Cap on the speculative margin used to size the candidate-cell window. The contact
        // distance gate still uses the full maxDistance; this only bounds the search volume.
        const float k_VoxelMaxSpeculativeMargin = 0.5f;

        // Masked direction components smaller than this count as zero (stabilizes the sign class
        // and avoids near-zero-length normal directions).
        const float k_VoxelSignDeadzone = 1e-4f;

        // Accumulated data of one (A voxel, sign class) contact bucket, in B grid space.
        struct VoxelContactBucket
        {
            public float Distance;    // most penetrating separation among all contributing B voxels
            public float3 NormalInB;  // unit normal (B grid), points from B towards A
            public float3 PosInB;     // contact position on B's surface
            public float3 CenterInB;  // A voxel center
        }

        struct VoxelEqualityPoint
        {
            public int Class;         // sign class of the visited bucket (axis classes merge per axis)
            public float Distance;    // (d_this - d_opposite) / 2: drives the voxel to the slack center
            public float3 NormalInB;
            public float3 PosInB;
        }

        // Trinary sign class of a masked direction: digit per axis in {0:-,1:zero,2:+},
        // class = dx*9 + dy*3 + dz. Class 13 is the all-zero direction (never stored).
        // The exactly opposite class is 26 - class.
        static int VoxelSignClass(float3 maskedDelta)
        {
            int dx = maskedDelta.x > k_VoxelSignDeadzone ? 2 : (maskedDelta.x < -k_VoxelSignDeadzone ? 0 : 1);
            int dy = maskedDelta.y > k_VoxelSignDeadzone ? 2 : (maskedDelta.y < -k_VoxelSignDeadzone ? 0 : 1);
            int dz = maskedDelta.z > k_VoxelSignDeadzone ? 2 : (maskedDelta.z < -k_VoxelSignDeadzone ? 0 : 1);
            return dx * 9 + dy * 3 + dz;
        }

        // Returns the axis (0..2) when the class has exactly one non-zero digit, else -1.
        static int VoxelClassAxis(int signClass)
        {
            int dx = signClass / 9;
            int dy = (signClass / 3) % 3;
            int dz = signClass % 3;
            int nonZero = (dx != 1 ? 1 : 0) + (dy != 1 ? 1 : 0) + (dz != 1 ? 1 : 0);
            if (nonZero != 1)
            {
                return -1;
            }

            return dx != 1 ? 0 : (dy != 1 ? 1 : 2);
        }

        // Packs a lattice coordinate plus sign class (0..26) into a single key.
        // Coordinates are biased into a 19 bit unsigned range (supports +-262143 blocks).
        static ulong PackVoxelClassKey(int3 coord, int signClass)
        {
            const int bias = 1 << 18;
            return (ulong)(uint)signClass
                | ((ulong)((uint)(coord.x + bias) & 0x7FFFF) << 5)
                | ((ulong)((uint)(coord.y + bias) & 0x7FFFF) << 24)
                | ((ulong)((uint)(coord.z + bias) & 0x7FFFF) << 43);
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

            // Conservative per-axis extent of a rotated A sector in B space (for culling only;
            // the sphere contact metric itself is rotation invariant).
            float3x3 rot = bFromA.Rotation;
            float3 rowAbsSum = math.abs(rot.c0) + math.abs(rot.c1) + math.abs(rot.c2);

            float speculative = math.clamp(maxDistance, 0.0f, k_VoxelMaxSpeculativeMargin);

            // Candidate cells must lie within sphere reach (center distance 1) plus margin per
            // axis. Keeping the window this tight is also what keeps the exposure-mask trick
            // local to face-adjacent surface planes.
            float windowHalfWidth = 1.0f + speculative;

            // Sphere-sphere contact gate on center distance, squared to avoid sqrt on misses.
            float maxCenterDistance = 1.0f + maxDistance;
            float maxCenterDistanceSq = maxCenterDistance * maxCenterDistance;

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

                                    float3 cellCenter = new float3(bx, by, bz) + 0.5f;
                                    float3 delta = centerLocalB - cellCenter;

                                    // Mask out directions whose B face is not exposed: flat
                                    // surfaces act flat (face-adjacent voxels share one surface
                                    // plane), interior faces can never push, and a fully interior
                                    // overlap produces no contact at all instead of a fake normal.
                                    for (int axis = 0; axis < 3; axis++)
                                    {
                                        float c = delta[axis];
                                        if (c > -k_VoxelSignDeadzone && c < k_VoxelSignDeadzone)
                                        {
                                            delta[axis] = 0.0f;
                                            continue;
                                        }

                                        int face = axis * 2 + (c > 0.0f ? 0 : 1);
                                        if ((infoB.data & (1 << face)) == 0)
                                        {
                                            delta[axis] = 0.0f;
                                        }
                                    }

                                    float distanceSq = math.lengthsq(delta);

                                    bool isPhysicsContact =
                                        distanceSq > 0.0f && distanceSq < maxCenterDistanceSq;
                                    float3 worldNormal = float3.zero;

                                    if (isPhysicsContact)
                                    {
                                        float centerDistance = math.sqrt(distanceSq);
                                        float3 normalInB = delta / centerDistance;
                                        float separation = centerDistance - 1.0f; // both radii 0.5

                                        worldNormal = math.mul(worldFromB.Rotation, normalInB);

                                        int signClass = VoxelSignClass(delta);
                                        int classAxis = VoxelClassAxis(signClass);

                                        // Axis-aligned contacts snap laterally to the A voxel
                                        // center on the B face plane (mergeable, no phantom
                                        // torque); diagonal contacts sit on the B sphere surface.
                                        float3 posInB;
                                        if (classAxis >= 0)
                                        {
                                            posInB = centerInB;
                                            posInB[classAxis] = sectorOriginB[classAxis] + cellCenter[classAxis]
                                                + (delta[classAxis] > 0.0f ? 0.5f : -0.5f);
                                        }
                                        else
                                        {
                                            posInB = (float3)sectorOriginB + cellCenter + 0.5f * normalInB;
                                        }

                                        ulong key = PackVoxelClassKey(voxelCoordA, signClass);
                                        if (buckets.TryGetValue(key, out VoxelContactBucket bucket))
                                        {
                                            if (separation < bucket.Distance)
                                            {
                                                bucket.Distance = separation;
                                                bucket.NormalInB = normalInB;
                                                bucket.PosInB = posInB;
                                                buckets[key] = bucket;
                                            }
                                        }
                                        else
                                        {
                                            buckets.Add(key, new VoxelContactBucket
                                            {
                                                Distance = separation,
                                                NormalInB = normalInB,
                                                PosInB = posInB,
                                                CenterInB = centerInB
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

        // Fuses opposing buckets into bilateral constraints and writes the remaining buckets as
        // unilateral contacts: one shared manifold per axis-aligned class, single-point manifolds
        // for diagonal (rounded corner/edge) classes.
        static unsafe void EmitVoxelContactManifolds(
            ref UnsafeHashMap<ulong, VoxelContactBucket> buckets,
            Context context,
            in MTransform worldFromB,
            Material materialA,
            Material materialB,
            bool flipped)
        {
            var kv = buckets.GetKeyValueArrays(Allocator.Temp);

            // Pass 1: fuse exactly opposing buckets on the same A voxel into single bilateral
            // points at the voxel center. Both sides must be touching within the slop: a loose
            // fit keeps rattling contacts, a deep squeeze keeps pushing contacts (anti-glue).
            var consumedKeys = new UnsafeHashSet<ulong>(16, Allocator.Temp);
            var equalityPoints = new UnsafeList<VoxelEqualityPoint>(16, Allocator.Temp);

            for (int i = 0; i < kv.Length; i++)
            {
                ulong key = kv.Keys[i];
                int signClass = (int)(key & 0x1F);
                if (signClass >= 13)
                {
                    continue; // visit the lower-class side only; the partner is looked up below
                }

                ulong oppositeKey = (key & ~(ulong)0x1F) | (uint)(26 - signClass);
                if (!buckets.TryGetValue(oppositeKey, out VoxelContactBucket opposite))
                {
                    continue;
                }

                VoxelContactBucket bucket = kv.Values[i];
                if (math.max(bucket.Distance, opposite.Distance) > k_VoxelEqualityMaxGap ||
                    math.min(bucket.Distance, opposite.Distance) < -k_VoxelEqualityMaxPenetration ||
                    math.dot(bucket.NormalInB, opposite.NormalInB) > k_VoxelEqualityMinOpposition)
                {
                    continue;
                }

                consumedKeys.Add(key);
                consumedKeys.Add(oppositeKey);
                equalityPoints.Add(new VoxelEqualityPoint
                {
                    Class = signClass,
                    // Drives the voxel towards the center of the remaining slack; both opposing
                    // sides agree on this target so they cannot fight each other.
                    Distance = 0.5f * (bucket.Distance - opposite.Distance),
                    NormalInB = bucket.NormalInB,
                    PosInB = bucket.CenterInB
                });
            }

            // Pass 2: bilateral manifolds. The three axis classes below 13 (4:-x, 10:-y, 12:-z)
            // have exact shared normals and merge into one manifold per axis; diagonal fusions
            // (pinched rounded corners, rare) are written individually. Equality points are load
            // bearing joints, so they are chunked rather than reduced when above the point limit.
            for (int axisClass = 0; axisClass < 13; axisClass++)
            {
                if (VoxelClassAxis(axisClass) < 0)
                {
                    continue;
                }

                var manifold = new ConvexConvexManifoldQueries.Manifold();
                bool normalSet = false;

                for (int i = 0; i < equalityPoints.Length; i++)
                {
                    if (equalityPoints[i].Class != axisClass)
                    {
                        continue;
                    }

                    if (!normalSet)
                    {
                        manifold.Normal = math.mul(worldFromB.Rotation, equalityPoints[i].NormalInB);
                        normalSet = true;
                    }

                    if (manifold.NumContacts == ConvexConvexManifoldQueries.Manifold.k_MaxNumContacts)
                    {
                        WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB,
                            flipped, JacobianFlags.IsBilateral);
                        manifold.NumContacts = 0;
                    }

                    manifold[manifold.NumContacts++] = new ContactPoint
                    {
                        Position = Mul(worldFromB, equalityPoints[i].PosInB),
                        Distance = equalityPoints[i].Distance
                    };

#if SHOW_DEBUG
                    Debug.DrawRay(Mul(worldFromB, equalityPoints[i].PosInB), manifold.Normal, Color.cyan, 0.0f, false);
#endif
                }

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB,
                    flipped, JacobianFlags.IsBilateral);
            }

            for (int i = 0; i < equalityPoints.Length; i++)
            {
                if (VoxelClassAxis(equalityPoints[i].Class) >= 0)
                {
                    continue; // already written above
                }

                var manifold = new ConvexConvexManifoldQueries.Manifold
                {
                    Normal = math.mul(worldFromB.Rotation, equalityPoints[i].NormalInB)
                };
                manifold[manifold.NumContacts++] = new ContactPoint
                {
                    Position = Mul(worldFromB, equalityPoints[i].PosInB),
                    Distance = equalityPoints[i].Distance
                };

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB,
                    flipped, JacobianFlags.IsBilateral);
            }

            // Pass 3: unilateral manifolds. Axis classes merge into one manifold per direction
            // (reduced to the support-polygon extremes plus the deepest points when above the
            // point limit); diagonal classes are written as single-point manifolds.
            var facePoints = new UnsafeList<VoxelContactBucket>(math.max(1, kv.Length), Allocator.Temp);
            for (int signClass = 0; signClass < 27; signClass++)
            {
                int classAxis = VoxelClassAxis(signClass);
                if (classAxis < 0)
                {
                    continue;
                }

                facePoints.Clear();
                for (int i = 0; i < kv.Length; i++)
                {
                    ulong key = kv.Keys[i];
                    if ((int)(key & 0x1F) == signClass && !consumedKeys.Contains(key))
                    {
                        facePoints.Add(kv.Values[i]);
                    }
                }

                if (facePoints.Length == 0)
                {
                    continue;
                }

                float3 normalInB = float3.zero;
                normalInB[classAxis] = signClass > 13 ? 1.0f : -1.0f;

                var manifold = new ConvexConvexManifoldQueries.Manifold
                {
                    Normal = math.mul(worldFromB.Rotation, normalInB)
                };

                if (facePoints.Length > ConvexConvexManifoldQueries.Manifold.k_MaxNumContacts)
                {
                    ReduceFacePoints(ref facePoints, classAxis, ConvexConvexManifoldQueries.Manifold.k_MaxNumContacts);
                }

                for (int i = 0; i < facePoints.Length; i++)
                {
                    manifold[manifold.NumContacts++] = new ContactPoint
                    {
                        Position = Mul(worldFromB, facePoints[i].PosInB),
                        Distance = facePoints[i].Distance
                    };

#if SHOW_DEBUG
                    Debug.DrawRay(Mul(worldFromB, facePoints[i].PosInB), manifold.Normal * 0.5f, Color.red, 0.0f, false);
#endif
                }

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
            }

            for (int i = 0; i < kv.Length; i++)
            {
                ulong key = kv.Keys[i];
                int signClass = (int)(key & 0x1F);
                if (VoxelClassAxis(signClass) >= 0 || consumedKeys.Contains(key))
                {
                    continue;
                }

                VoxelContactBucket bucket = kv.Values[i];
                var manifold = new ConvexConvexManifoldQueries.Manifold
                {
                    Normal = math.mul(worldFromB.Rotation, bucket.NormalInB)
                };
                manifold[manifold.NumContacts++] = new ContactPoint
                {
                    Position = Mul(worldFromB, bucket.PosInB),
                    Distance = bucket.Distance
                };

#if SHOW_DEBUG
                Debug.DrawRay(Mul(worldFromB, bucket.PosInB), manifold.Normal * 0.5f, Color.yellow, 0.0f, false);
#endif

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
            }

            facePoints.Dispose();
            equalityPoints.Dispose();
            consumedKeys.Dispose();
            kv.Dispose();
        }

        // In-place selection of up to maxPoints from an axis-class point set: the 8 lateral
        // extreme points (the support polygon rim, which gives rocking stability) plus the
        // deepest of the rest. Order within the list is not meaningful.
        static void ReduceFacePoints(ref UnsafeList<VoxelContactBucket> points, int axis, int maxPoints)
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
                    float score = points[i].PosInB[u] * du + points[i].PosInB[v] * dv;
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
