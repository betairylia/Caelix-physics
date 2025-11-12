// #define SHOW_DEBUG

using Unity.Physics;
using Unity.Mathematics;
using UnityEngine;
using Voxelis;
using static Unity.Physics.Math;

namespace Unity.Physics
{
    static partial class ManifoldQueries
    {
        internal static unsafe void VoxelVoxel(
            Context context,
            Collider* colliderA,
            Collider* colliderB,
            Math.MTransform worldFromA,
            Math.MTransform worldFromB,
            float maxDistance,
            bool flipped)
        {
            const float EPSILON = 1e-5f;

            VoxelCollider* voxelA = (VoxelCollider*)colliderA;
            VoxelCollider* voxelB = (VoxelCollider*)colliderB;

            // Get materials (use default for voxels since they don't have material headers)
            Material materialA = Material.Default;
            Material materialB = Material.Default;

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

            // Calculate transformation from A to B space
            Math.MTransform bFromA = Math.Mul(Math.Inverse(worldFromB), worldFromA);

            // Get the sector maps
            var sectorsA = voxelA->m_Sectors;
            var sectorsB = voxelB->m_Sectors;

            if (!sectorsA.IsCreated || !sectorsB.IsCreated || sectorsA.IsEmpty || sectorsB.IsEmpty)
            {
                return;
            }

            // Iterate through all sectors in A
            var keysA = sectorsA.GetKeyArray(Unity.Collections.Allocator.Temp);

            for (int iSectorA = 0; iSectorA < keysA.Length; iSectorA++)
            {
                int3 sectorCoordA = keysA[iSectorA];
                var sectorA = sectorsA[sectorCoordA];

                // For each block in sector A, we need to check which sector(s) in B it might collide with
                // For simplicity in this prototype, we'll check all sectors in B
                var keysB = sectorsB.GetKeyArray(Unity.Collections.Allocator.Temp);

                for (int iSectorB = 0; iSectorB < keysB.Length; iSectorB++)
                {
                    int3 sectorCoordB = keysB[iSectorB];
                    var sectorB = sectorsB[sectorCoordB];

                    // Create a manifold for this sector pair
                    var manifold = new ConvexConvexManifoldQueries.Manifold();

                    // Iterate through all non-empty blocks in sector A
                    foreach (Voxelis.BlockIterator blockIter in new Voxelis.SectorEnumerator(*sectorA.Ptr))
                    {
                        // Calculate the world position of the block in A
                        float3 srcBlockWorldPos = new float3(blockIter.position + sectorCoordA * Voxelis.Sector.SECTOR_SIZE_IN_BLOCKS);
                        float3 srcBlockCenter = srcBlockWorldPos + 0.5f;

                        // Transform to B space
                        float3 dstBlockCenter = Math.Mul(bFromA, srcBlockCenter);

                        // Adjust for sector B's coordinate system
                        float3 dstBlockLocal = dstBlockCenter - (float3)(sectorCoordB * Voxelis.Sector.SECTOR_SIZE_IN_BLOCKS);
                        float3 dstBlockOrigin = dstBlockLocal - 0.5f + EPSILON;

                        // Calculate which blocks to check (up to 2x2x2 grid)
                        int3 dstBlock2x2x2Origin = new int3(dstBlockOrigin);
                        int3 exact = new int3((dstBlockOrigin - (float3)dstBlock2x2x2Origin) < (2 * EPSILON));

                        for (int dx = 0; dx < 2 - exact.x; dx++)
                        {
                            for (int dy = 0; dy < 2 - exact.y; dy++)
                            {
                                for (int dz = 0; dz < 2 - exact.z; dz++)
                                {
                                    int3 destination = new int3(
                                        dstBlock2x2x2Origin.x + dx,
                                        dstBlock2x2x2Origin.y + dy,
                                        dstBlock2x2x2Origin.z + dz
                                    );

                                    // Check out-of-bounds
                                    if (math.any(destination < 0) ||
                                        math.any(destination >= new int3(
                                            Voxelis.Sector.SECTOR_SIZE_IN_BLOCKS,
                                            Voxelis.Sector.SECTOR_SIZE_IN_BLOCKS,
                                            Voxelis.Sector.SECTOR_SIZE_IN_BLOCKS
                                        )))
                                    {
                                        continue;
                                    }

                                    var dstBlock = sectorB.GetBlock(
                                        destination.x, destination.y, destination.z
                                    );

                                    if (dstBlock.isEmpty)
                                    {
                                        continue;
                                    }

                                    float3 dstTargetCenter = (float3)destination + 0.5f;

                                    // Basic sphere-ish collision (same as SectorJob)
                                    float dsq = math.lengthsq(dstBlockLocal - dstTargetCenter);
                                    if (dsq <= 0.9216f)
                                    {
                                        // Prototype: Use SphereSphere directly
                                        DistanceQueries.Result convexDistance = DistanceQueries.PointPoint(
                                            dstBlockLocal + sectorCoordB * Sector.SECTOR_SIZE_IN_BLOCKS,
                                            dstTargetCenter + sectorCoordB * Sector.SECTOR_SIZE_IN_BLOCKS,
                                            0.48f, 0.96f
                                        );
                                        manifold = new ConvexConvexManifoldQueries.Manifold(convexDistance, worldFromB);
                                        WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);

                                        // Write voxel contact event data
                                        // Calculate voxel coordinates (sectorPos * SECTOR_SIZE_IN_BLOCKS + in-sector-coord)
                                        int3 voxelCoordsInA = sectorCoordA * Voxelis.Sector.SECTOR_SIZE_IN_BLOCKS + blockIter.position;
                                        int3 voxelCoordsInB = sectorCoordB * Voxelis.Sector.SECTOR_SIZE_IN_BLOCKS + destination;

                                        // Normal is A to B (manifold.Normal is already correct after flip handling in WriteManifold)
                                        var voxelContactData = new VoxelContactEventData
                                        {
                                            BodyIndices = context.BodyIndices,
                                            VoxelCoordsInA = flipped ? voxelCoordsInB : voxelCoordsInA,
                                            VoxelCoordsInB = flipped ? voxelCoordsInA : voxelCoordsInB,
                                            Normal = manifold.Normal
                                        };

                                        context.VoxelContactWriter->Write(voxelContactData);
                                        
#if SHOW_DEBUG
                                        Debug.DrawLine(manifold[0].Position, manifold[0].Position + manifold.Normal * manifold[0].Distance * 100.0f, Color.red, 0.0f, true);
                                        Debug.Log($"Contact Manifold: {manifold[0].Position} -> {manifold[0].Position + manifold.Normal * manifold[0].Distance}");
#endif
                                        
                                        // Calculate contact in B space then transform to world space
                                        // float3 contactPosB = (dstBlockLocal + dstTargetCenter) / 2.0f + (float3)(sectorCoordB * Voxelis.Sector.SECTOR_SIZE_IN_BLOCKS);
                                        // float3 normalB = math.normalize(dstTargetCenter - dstBlockLocal);
                                        //
                                        // // Transform contact position to world space
                                        // float3 contactPosWorld = Math.Mul(worldFromB, contactPosB);
                                        // float3 normalWorld = math.mul(worldFromB.Rotation, normalB);
                                        //
                                        // float distance = math.sqrt(dsq) - 1.0f;
                                        //
                                        // // Add to manifold or flush if different normal
                                        // // if (manifold.NumContacts > 0 && math.dot(manifold.Normal, normalWorld) < 0.99f)
                                        // if (manifold.NumContacts > 0)
                                        // {
                                        //     // Flush manifold with current normal
                                        //     WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
                                        //     manifold = new ConvexConvexManifoldQueries.Manifold
                                        //     {
                                        //         Normal = normalWorld
                                        //     };
                                        // }
                                        // else if (manifold.NumContacts == 0)
                                        // {
                                        //     manifold.Normal = normalWorld;
                                        // }
                                        //
                                        // if (manifold.NumContacts < ConvexConvexManifoldQueries.Manifold.k_MaxNumContacts)
                                        // {
                                        //     manifold[manifold.NumContacts++] = new ContactPoint
                                        //     {
                                        //         Position = contactPosWorld,
                                        //         Distance = distance
                                        //     };
                                        // }
                                        // else
                                        // {
                                        //     // Flush and start new manifold
                                        //     WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
                                        //     manifold = new ConvexConvexManifoldQueries.Manifold
                                        //     {
                                        //         Normal = normalWorld
                                        //     };
                                        //     manifold[manifold.NumContacts++] = new ContactPoint
                                        //     {
                                        //         Position = contactPosWorld,
                                        //         Distance = distance
                                        //     };
                                        // }
                                    }
                                }
                            }
                        }
                    }

                    // Flush any remaining contacts for this sector pair
                    if (manifold.NumContacts > 0)
                    {
                        WriteManifold(manifold, context, ColliderKeyPair.Empty, materialA, materialB, flipped);
                    }
                }

                keysB.Dispose();
            }

            keysA.Dispose();
        }
    }
}