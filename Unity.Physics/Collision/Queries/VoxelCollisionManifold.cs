#define SHOW_DEBUG

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Voxelis;
using static Unity.Physics.Math;
using System;
using NUnit.Framework;



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
        // Masking an axis is only valid inside the B cell's own transverse footprint: a face
        // plane extends across face-adjacent solid neighbors, but each cell only owns the part
        // over its own footprint. When A lies beyond the footprint on an unexposed axis, the
        // neighbor cell on that side owns the surface facing A and this cell emits nothing.
        // (Without this gate, inside corners fabricate full-depth phantom contacts: the masked
        // delta of a diagonally offset A voxel collapses to a near-zero residual, reporting
        // separation ~ -1 along an arbitrary exposed axis.)
        //
        // This file intentionally contains GENERATION ONLY: every surviving (A voxel, B cell)
        // probe emits one raw contact, written as its own single-point manifold plus one contact
        // event. There is no deduplication, merging, or reduction - a voxel straddling a cell
        // seam emits one contact per facing cell (identical plane contacts). The previous
        // merging / joint extraction stages (per-(A voxel, sign class) keep-deepest buckets,
        // fusion of opposing contacts into bilateral equality points, per-axis manifold merging,
        // rim/extreme reduction) were removed 2026-07 to make room for a new patch-based merging
        // algorithm; see git history for the old implementation.
        //
        // Conventions (matching the rest of the narrowphase):
        //   * Manifold normals point from B towards A; positive contact distance is separation.
        //   * Positions are in world space, on body B's surface.
        //   * PhysicsInfo exposure bits follow NeighborhoodSettings face order:
        //     bit = axis * 2 + (0 for +axis, 1 for -axis); a set bit means the face is exposed.
        //
        // Uniform body scale is not supported here (assumed 1), same as the previous prototype.
        // ---------------------------------------------------------------------------------------

        // Cap on the speculative margin used to size the candidate-cell window. The contact
        // distance gate still uses the full maxDistance; this only bounds the search volume.
        const float k_VoxelMaxSpeculativeMargin = 0.5f;

        // Masked direction components smaller than this count as zero (avoids near-zero-length
        // normal directions).
        const float k_VoxelSignDeadzone = 1e-4f;

        // Half width of the transverse footprint a B cell owns on a masked (unexposed) axis:
        // half a cell plus a small seam margin so contacts stay continuous while A crosses cell
        // boundaries (both cells emit the same plane contact inside the margin band).
        const float k_VoxelFootprintHalfWidth = 0.55f;

        const float k_VoxelConstraintEdgeEdgeDegenerationThreshold = 0.9f;

        // One raw generated contact, in B grid space.
        struct VoxelContact
        {
            public float Distance;    // separation along the normal (negative = penetration)
            public float3 NormalInB;  // unit normal (B grid), points from B towards A
            public float3 PosInB;     // contact position on B's surface
            public int3 VoxelA;       // contributing A voxel (A grid)
            public int3 VoxelB;       // contributing B voxel (B grid)
            // public bool Diagonal;     // corner/edge rounding contact (not an axis-face contact)

            // Bin assignment of this contact point.
            // Contact points are binned by their normal directions.
            // The masking step computes corrected normals and bin assignments from raw normals (raw Avoxel-Bvoxel center differences) and face masks (PhysicsInfo slot).
            // Let the rank of a contact be the number of constraints the contact normal must satisfy. 1 unexposed face corresponds to 1 constraint (dot-product = 0).
            // Bins are arranged by:
            //   0     - Rank-0 (Corner-Corner) or Rank-1 (Corner-Edge) contacts
            //   1-3   - Constraint on Entity A's YZ, XZ or XY axes (i.e., normal is A's X, Y and Z)
            //   4-6   - Constraint on Entity B's YZ, XZ or XY axes (i.e., normal is B's X, Y and Z)
            //   7-15  - Constraint on {Entity A's X, Y, Z} x {Entity B's X, Y, Z} axes (i.e., normal is computed via cross-product)
            // Negative bin value is possible and it represents negative direction in the corresponding constrainted axis.
            // In following stages, bin info will be used to further merge contacts.
            // "Rank-3 and Rank-4" contacts should be omitted / ignored and not creating any contacts. Instead, the solver should rely on Rank-0/1/2 contacts only.
            public int normalBin;
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

            if (!voxelA->m_Sectors.IsCreated || !voxelB->m_Sectors.IsCreated ||
                voxelA->m_Sectors.IsEmpty || voxelB->m_Sectors.IsEmpty)
            {
                return;
            }

            MTransform bFromA = Mul(Inverse(worldFromB), worldFromA);

            var contacts = new UnsafeList<VoxelContact>(256, Allocator.Temp);
            CollectVoxelContacts(voxelA, voxelB, bFromA, maxDistance, ref contacts);
            WriteVoxelManifolds(contacts, context, worldFromB, materialA, materialB, flipped);
            contacts.Dispose();
        }

        private static bool ApplyNormalMasking(
            float3 delta, PhysicsInfo infoA, PhysicsInfo infoB, MTransform bFromA,
            out float3 maskedDelta, out int normalBin)
        {
            maskedDelta = delta;

            // Flags bits store 3 - (solid-surrounded axis count); ~ over the int-promoted
            // byte sets bits 8..31, so mask back down to the 2-bit field.
            // TODO: FIXME: This does not align with the true constraint rank when half-constraints exists.
            // But, How do we correctly count number of constraints for half-constraints (boundary voxels)?
            int numConstraintsA_lowerBound = ((~infoA.data) >> 6) & 0b11;
            int numConstraintsB_lowerBound = ((~infoB.data) >> 6) & 0b11;
            int numConstraintsTotal_lowerBound = numConstraintsA_lowerBound + numConstraintsB_lowerBound;

            // TODO: Skip some corner-corner cases where (data & 0x3F) == 0x3F?

            // TODO: Check degenerate case?
            // Allow only CORNER-FACE (0+2) & EDGE-EDGE (1+1) pairs.
            if (numConstraintsTotal_lowerBound > 2)
            {
                normalBin = 0;
                return false;
            }

            // Find constraint axes
            float3 constraintA = float3.zero, constraintB = float3.zero;

            int numConstraintsA = 0, numConstraintsB = 0;
            int _debug_constraintRecord = 0;

            // Start from B, then A; B's constraints takes priority.
            // We cannot determine #constraints before checking the side of axis (computing `c`).
            // Furthermore, even if we may have more than 2 constarints, we cannot simply drop e.g., an EDGE-EDGE
            // with 3 constraints since there will be no fallback neighbors to catch the actual collision.
            // To handle this case, we priortize B's constraints rather than drop rank>2 collisions.
            for (int obj = 1; obj >= 0; obj--)
            {
                for (int axis = 0; axis < 3; axis++)
                {
                    // A's axes expressed in B space are the columns of bFromA.Rotation
                    // (v_B = Rotation * v_A); B's axes in B space are the identity basis.
                    float3 constraintAxis;
                    int dataToRef = infoA.data;
                    int axisID = obj * 3 + axis;
                    switch (axisID)
                    {
                        case 0: // A's X
                            constraintAxis = bFromA.Rotation[0];
                            break;
                        case 1: // A's Y
                            constraintAxis = bFromA.Rotation[1];
                            break;
                        case 2: // A's Z
                            constraintAxis = bFromA.Rotation[2];
                            break;
                        case 3: // B's X
                            constraintAxis = new float3(1, 0, 0);
                            dataToRef = infoB.data;
                            break;
                        case 4: // B's Y
                            constraintAxis = new float3(0, 1, 0);
                            dataToRef = infoB.data;
                            break;
                        default: // 5: B's Z
                            constraintAxis = new float3(0, 0, 1);
                            dataToRef = infoB.data;
                            break;
                    }

                    float c = math.dot(obj == 0 ? (-delta) : delta, constraintAxis);
                    int face = axis * 2 + (c > 0.0f ? 0 : 1);

                    // Mask out directions whose A or B face is not exposed: flat
                    // surfaces act flat (face-adjacent voxels share one surface
                    // plane), interior faces can never push, and a fully interior
                    // overlap produces no contact at all instead of a fake normal.
                    // Masking is footprint-gated: beyond the cell's own footprint
                    // on an unexposed axis, the neighbor owns the surface facing
                    // A and this cell emits nothing (see header).
                    if ((dataToRef & (1 << face)) == 0)
                    {
                        // Skip phantom contacts
                        if (c > k_VoxelFootprintHalfWidth || c < -k_VoxelFootprintHalfWidth)
                        {
                            normalBin = 0;
                            return false;
                        }

                        _debug_constraintRecord += axisID * (int)math.exp10(numConstraintsA + numConstraintsB);

                        switch (obj)
                        {
                            case 0:
                                if (numConstraintsA == 0) { constraintA = constraintAxis; }
                                else { constraintA = math.cross(constraintA, constraintAxis); }
                                numConstraintsA++;
                                break;
                            case 1:
                                if (numConstraintsB == 0) { constraintB = constraintAxis; }
                                else { constraintB = math.cross(constraintB, constraintAxis); }
                                numConstraintsB++;
                                break;
                        }
                    }
                }

                // We already found enough constraints
                if (numConstraintsB == 2)
                {
                    break;
                }
            }

            // Interior. How can this happen?
            if (numConstraintsA > 2 || numConstraintsB > 2)
            {
                normalBin = 0;
                return false;
            }

            //////////////////////////////////////
            /// Pick the correct constraint set
            //////////////////////////////////////

            float3 constraint = float3.zero;
            int numConstraintsTotal = 0;

            // Case. Clean edge-to-edge
            if (numConstraintsA == 1 && numConstraintsB == 1)
            {
                // Degenerate case
                if(math.abs(math.dot(constraintA, constraintB)) > k_VoxelConstraintEdgeEdgeDegenerationThreshold)
                {
                    constraint = constraintB;
                    numConstraintsTotal = numConstraintsB;
                }
                else
                {
                    constraint = math.normalize(math.cross(constraintB, constraintA));
                    numConstraintsTotal = 2;
                }
            }
            // Case. Use A
            else if (numConstraintsA > 0 && numConstraintsB < 2)
            {
                constraint = constraintA;
                numConstraintsTotal = numConstraintsA;
            }
            // Case. Use B
            else if (numConstraintsB > 0 && numConstraintsA < 2)
            {
                constraint = constraintB;
                numConstraintsTotal = numConstraintsB;
            }
            // Case. No constraints
            else if (numConstraintsA == 0 && numConstraintsB == 0)
            {
                normalBin = 0;
                return true;
            }
            // We should not reach this
            else
            {
                throw new Exception("Bad contact constraints. How possibly can this happen?");
            }

            ///////////////////////////
            /// Apply constraints
            ///////////////////////////
            
            if (numConstraintsTotal == 0)
            {
                // No constraints in the half of delta, do nothing
                normalBin = 0;
            }
            else if (numConstraintsTotal == 1)
            {
                maskedDelta = maskedDelta - math.dot(delta, constraint) * constraint;
                // TODO: How to compute bin?
                normalBin = 0;
            }
            else
            {
                maskedDelta = math.dot(delta, constraint) * constraint;
                // if (math.abs(math.dot(delta, constraint)) < 0.5f)
                // {
                //     Debug.Log("!");
                //     Debug.Log($"Delta: {delta} | Constraint: {constraint} => Masked: {maskedDelta}");
                // }
                // TODO: How to compute bin?
                normalBin = 0;
            }

            normalBin = _debug_constraintRecord;
            return true;
        }

        // For every exposed A voxel, probe the B cells within sphere reach, mask the
        // center-to-center direction by B's face exposure (footprint-gated, see header), and
        // append one raw contact per surviving probe.
        static unsafe void CollectVoxelContacts(
            VoxelCollider* voxelA,
            VoxelCollider* voxelB,
            in MTransform bFromA,
            float maxDistance,
            ref UnsafeList<VoxelContact> contacts)
        {
            var sectorsA = voxelA->m_Sectors;
            var sectorsB = voxelB->m_Sectors;

            // Conservative per-axis extent of a rotated A sector in B space (for culling only;
            // the sphere contact metric itself is rotation invariant).
            float3x3 rot = bFromA.Rotation;
            float3 rowAbsSum = math.abs(rot.c0) + math.abs(rot.c1) + math.abs(rot.c2);

            float speculative = math.clamp(maxDistance, 0.0f, k_VoxelMaxSpeculativeMargin);

            // TODO: Check this thing's tightness.
            // Candidate cells must lie within sphere reach (center distance 1) plus margin per
            // axis. Keeping the window this tight is also what keeps the exposure-mask trick
            // local to face-adjacent surface planes.
            float windowHalfWidth = 1.0f + speculative;

            // Sphere-sphere contact gate on center distance, squared to avoid sqrt on misses.
            float maxCenterDistance = 1.0f + maxDistance;
            float maxCenterDistanceSq = maxCenterDistance * maxCenterDistance;

            var keysA = sectorsA.GetKeyArray(Allocator.Temp);
            var keysB = sectorsB.GetKeyArray(Allocator.Temp);

            // TODO: Maybe optimize this nested O(N_sector_A * N_sector_B) loop? Perhaps this is not needed.
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

                        // TODO: Check this thing's tightness.
                        // TODO: FIXME: The lo-hi AABB is too loose to ensure interior voxels are not visited
                        // in below loop, and renders the necessity of ownedByNeighbor. It should be possible
                        // to completely remove this additional flag for simplicity and correctness.

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

                                    // Apply normal masking and filter phantoms & face-edge & face-face
                                    bool shouldProceed = ApplyNormalMasking(delta, infoA, infoB, bFromA, out float3 maskedDelta, out int normalBin);

                                    if (!shouldProceed)
                                    {
                                        continue;
                                    }

                                    // Check distance

                                    float distanceSq = math.lengthsq(maskedDelta);
                                    if (distanceSq <= 0.0f || distanceSq >= maxCenterDistanceSq)
                                    {
                                        continue;
                                    }

                                    float centerDistance = math.sqrt(distanceSq);
                                    float3 normalInB = maskedDelta / centerDistance;
                                    float separation = centerDistance - 1.0f; // both radii 0.5

                                    // Exactly one non-zero component left after masking makes an
                                    // axis-face contact (masked components are exactly zero).
                                    int faceAxis = -1;
                                    int nonZero = 0;
                                    for (int axis = 0; axis < 3; axis++)
                                    {
                                        if (maskedDelta[axis] != 0.0f)
                                        {
                                            nonZero++;
                                            faceAxis = axis;
                                        }
                                    }

                                    if (nonZero != 1)
                                    {
                                        faceAxis = -1;
                                    }

                                    // TODO: Do we really need this snap?
                                    // Axis-aligned contacts snap laterally to the A voxel center
                                    // on the B face plane (no phantom torque); diagonal contacts
                                    // sit on the B sphere surface.
                                    float3 posInB;
                                    if (faceAxis >= 0)
                                    {
                                        posInB = centerInB;
                                        posInB[faceAxis] = sectorOriginB[faceAxis] + cellCenter[faceAxis]
                                            + (maskedDelta[faceAxis] > 0.0f ? 0.5f : -0.5f);
                                    }
                                    else
                                    {
                                        posInB = (float3)sectorOriginB + cellCenter + 0.5f * normalInB;
                                    }

                                    contacts.Add(new VoxelContact
                                    {
                                        Distance = separation,
                                        NormalInB = normalInB,
                                        PosInB = posInB,
                                        VoxelA = voxelCoordA,
                                        VoxelB = sectorOriginB + new int3(bx, by, bz),
                                        normalBin = normalBin
                                        // Diagonal = faceAxis < 0
                                    });
                                }
                            }
                        }
                    }
                }
            }

            keysA.Dispose();
            keysB.Dispose();
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
                    Normal = math.mul(worldFromB.Rotation, contact.NormalInB),
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
                Debug.DrawRay(Mul(worldFromB, contact.PosInB), manifold.Normal * 0.5f,
                    Color.red, 0.0f, false);
#endif

                WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
            }
        }
    }
}
