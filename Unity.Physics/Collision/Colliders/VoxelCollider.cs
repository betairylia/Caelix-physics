using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Mathematics;
using Unity.VisualScripting.YamlDotNet.Core.Tokens;
using UnityEngine;
using Voxelis;

namespace Unity.Physics
{
    public unsafe struct VoxelCollider : ICollider
    {
        public UnsafeHashMap<int3, Sector> m_Sectors;
        public CollisionFilter m_Filter;

        /// <summary>
        /// Calculate the axis-aligned bounding box of this voxel collider in local space.
        /// Returns an AABB that encompasses all sectors, treating each sector as a 128x128x128 box.
        /// </summary>
        /// <returns>The local-space AABB covering all sectors, or an empty AABB if no sectors exist.</returns>
        public Aabb CalculateAabb()
        {
            return CalculateAabb(RigidTransform.identity);
        }

        /// <summary>
        /// Calculate the axis-aligned bounding box of this voxel collider transformed by the given rigid transform.
        /// This properly handles rotation by transforming the corners of the local AABB.
        /// </summary>
        /// <param name="transform">The rigid transform to apply (rotation and translation).</param>
        /// <param name="uniformScale">Uniform scale factor to apply (default: 1.0).</param>
        /// <returns>The transformed AABB covering all sectors.</returns>
        public Aabb CalculateAabb(RigidTransform transform, float uniformScale = 1.0f)
        {
            // If no sectors exist, return an empty AABB
            if (!m_Sectors.IsCreated || m_Sectors.IsEmpty)
            {
                return Aabb.Empty;
            }

            // Start with an empty AABB and expand it to include all sectors
            Aabb localAabb = Aabb.Empty;

            // Iterate through all sectors and union their AABBs
            var keys = m_Sectors.GetKeyArray(Allocator.Temp);
            for (int i = 0; i < keys.Length; i++)
            {
                int3 sectorCoord = keys[i];

                // Each sector is 128x128x128 blocks in size
                // Calculate the min and max corners of this sector in local space
                float3 sectorMin = sectorCoord * Sector.SECTOR_SIZE_IN_BLOCKS;
                float3 sectorMax = (sectorCoord + 1) * Sector.SECTOR_SIZE_IN_BLOCKS;

                // Expand the overall AABB to include this sector
                localAabb.Include(sectorMin);
                localAabb.Include(sectorMax);
            }
            keys.Dispose();

            // Transform the local AABB by the given transform and scale
            // This handles rotation properly by transforming the corner points
            return Math.TransformAabb(localAabb, transform, uniformScale);
        }

        #region Query Methods
        
        public bool CastRay(RaycastInput input)
        {
            throw new System.NotImplementedException();
        }

        public bool CastRay(RaycastInput input, out RaycastHit closestHit)
        {
            throw new System.NotImplementedException();
        }

        public bool CastRay(RaycastInput input, ref NativeList<RaycastHit> allHits)
        {
            throw new System.NotImplementedException();
        }

        public bool CastRay<T>(RaycastInput input, ref T collector) where T : struct, ICollector<RaycastHit>
        {
            throw new System.NotImplementedException();
        }

        public bool CastCollider(ColliderCastInput input)
        {
            throw new System.NotImplementedException();
        }

        public bool CastCollider(ColliderCastInput input, out ColliderCastHit closestHit)
        {
            throw new System.NotImplementedException();
        }

        public bool CastCollider(ColliderCastInput input, ref NativeList<ColliderCastHit> allHits)
        {
            throw new System.NotImplementedException();
        }

        public bool CastCollider<T>(ColliderCastInput input, ref T collector) where T : struct, ICollector<ColliderCastHit>
        {
            throw new System.NotImplementedException();
        }

        public bool CalculateDistance(PointDistanceInput input)
        {
            throw new System.NotImplementedException();
        }

        public bool CalculateDistance(PointDistanceInput input, out DistanceHit closestHit)
        {
            throw new System.NotImplementedException();
        }

        public bool CalculateDistance(PointDistanceInput input, ref NativeList<DistanceHit> allHits)
        {
            throw new System.NotImplementedException();
        }

        public bool CalculateDistance<T>(PointDistanceInput input, ref T collector) where T : struct, ICollector<DistanceHit>
        {
            throw new System.NotImplementedException();
        }

        public bool CalculateDistance(ColliderDistanceInput input)
        {
            throw new System.NotImplementedException();
        }

        public bool CalculateDistance(ColliderDistanceInput input, out DistanceHit closestHit)
        {
            throw new System.NotImplementedException();
        }

        public bool CalculateDistance(ColliderDistanceInput input, ref NativeList<DistanceHit> allHits)
        {
            throw new System.NotImplementedException();
        }

        public bool CalculateDistance<T>(ColliderDistanceInput input, ref T collector) where T : struct, ICollector<DistanceHit>
        {
            throw new System.NotImplementedException();
        }

        public bool CheckSphere(float3 position, float radius, CollisionFilter filter,
            QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool OverlapSphere(float3 position, float radius, ref NativeList<DistanceHit> outHits, CollisionFilter filter,
            QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool OverlapSphereCustom<T>(float3 position, float radius, ref T collector, CollisionFilter filter,
            QueryInteraction queryInteraction = QueryInteraction.Default) where T : struct, ICollector<DistanceHit>
        {
            throw new System.NotImplementedException();
        }

        public bool CheckCapsule(float3 point1, float3 point2, float radius, CollisionFilter filter,
            QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool OverlapCapsule(float3 point1, float3 point2, float radius, ref NativeList<DistanceHit> outHits, CollisionFilter filter,
            QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool OverlapCapsuleCustom<T>(float3 point1, float3 point2, float radius, ref T collector, CollisionFilter filter,
            QueryInteraction queryInteraction = QueryInteraction.Default) where T : struct, ICollector<DistanceHit>
        {
            throw new System.NotImplementedException();
        }

        public bool CheckBox(float3 center, quaternion orientation, float3 halfExtents, CollisionFilter filter,
            QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool OverlapBox(float3 center, quaternion orientation, float3 halfExtents, ref NativeList<DistanceHit> outHits,
            CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool OverlapBoxCustom<T>(float3 center, quaternion orientation, float3 halfExtents, ref T collector,
            CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default) where T : struct, ICollector<DistanceHit>
        {
            throw new System.NotImplementedException();
        }

        public bool SphereCast(float3 origin, float radius, float3 direction, float maxDistance, CollisionFilter filter,
            QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool SphereCast(float3 origin, float radius, float3 direction, float maxDistance, out ColliderCastHit hitInfo,
            CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool SphereCastAll(float3 origin, float radius, float3 direction, float maxDistance, ref NativeList<ColliderCastHit> outHits,
            CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool SphereCastCustom<T>(float3 origin, float radius, float3 direction, float maxDistance, ref T collector,
            CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default) where T : struct, ICollector<ColliderCastHit>
        {
            throw new System.NotImplementedException();
        }

        public bool BoxCast(float3 center, quaternion orientation, float3 halfExtents, float3 direction, float maxDistance,
            CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool BoxCast(float3 center, quaternion orientation, float3 halfExtents, float3 direction, float maxDistance,
            out ColliderCastHit hitInfo, CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool BoxCastAll(float3 center, quaternion orientation, float3 halfExtents, float3 direction, float maxDistance,
            ref NativeList<ColliderCastHit> outHits, CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool BoxCastCustom<T>(float3 center, quaternion orientation, float3 halfExtents, float3 direction, float maxDistance,
            ref T collector, CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default) where T : struct, ICollector<ColliderCastHit>
        {
            throw new System.NotImplementedException();
        }

        public bool CapsuleCast(float3 point1, float3 point2, float radius, float3 direction, float maxDistance,
            CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool CapsuleCast(float3 point1, float3 point2, float radius, float3 direction, float maxDistance,
            out ColliderCastHit hitInfo, CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool CapsuleCastAll(float3 point1, float3 point2, float radius, float3 direction, float maxDistance,
            ref NativeList<ColliderCastHit> outHits, CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default)
        {
            throw new System.NotImplementedException();
        }

        public bool CapsuleCastCustom<T>(float3 point1, float3 point2, float radius, float3 direction, float maxDistance,
            ref T collector, CollisionFilter filter, QueryInteraction queryInteraction = QueryInteraction.Default) where T : struct, ICollector<ColliderCastHit>
        {
            throw new System.NotImplementedException();
        }
        
        #endregion
        
        public ColliderType Type { get => ColliderType.Voxel; }
        public CollisionType CollisionType { get => CollisionType.Voxel; }
        public MassProperties MassProperties { get; }
        public int MemorySize { get; }
        
        public CollisionFilter GetCollisionFilter()
        {
            return m_Filter;
        }

        public void SetCollisionFilter(CollisionFilter filter)
        {
            m_Filter = filter;
        }

        public void BakeTransform(AffineTransform transform)
        {
            SafetyChecks.LogWarning($"Not implemented for collider type {Type}.");
        }

        public static BlobAssetReference<VoxelCollider> Create(
            IDictionary<Vector3Int, Sector> sectorMap,
            CollisionFilter filter, Material material)
        {
            unsafe
            {
                var collider = default(VoxelCollider);
                collider.m_Sectors = new UnsafeHashMap<int3, Sector>(sectorMap.Count, Allocator.Persistent);

                foreach (var kvp in sectorMap)
                {
                    collider.m_Sectors.Add(
                        new int3(kvp.Key.x, kvp.Key.y, kvp.Key.z), kvp.Value);
                }

                return BlobAssetReference<VoxelCollider>.Create(&collider, sizeof(VoxelCollider));
            }
        }
    }
}