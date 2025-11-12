using System;
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
        // Header
        private ColliderHeader m_Header;

        /// <summary>   The material. </summary>
        public Material Material;

        // Mass properties (can be set externally via SetMassProperties)
        private MassProperties m_MassProperties;

        public UnsafeHashMap<int3, SectorHandle> m_Sectors;

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
        
        public ColliderType Type => m_Header.Type;
        public CollisionType CollisionType => m_Header.CollisionType;

        /// <summary>
        /// Gets the mass properties. This should be set using SetMassProperties()
        /// after computing via VoxelBody.UpdateBody() or similar methods.
        /// </summary>
        public MassProperties MassProperties => m_MassProperties;

        /// <summary>
        /// Sets the mass properties for this voxel collider.
        /// Use this to apply computed mass properties from VoxelBody.UpdateBody() or other sources.
        /// </summary>
        /// <param name="massProperties">The computed mass properties to apply.</param>
        public void SetMassProperties(MassProperties massProperties)
        {
            m_MassProperties = massProperties;
            m_Header.Version++;
        }

        public int MemorySize => UnsafeUtility.SizeOf<VoxelCollider>();
        
        public CollisionFilter GetCollisionFilter()
        {
            return m_Header.Filter;
        }

        public void SetCollisionFilter(CollisionFilter filter)
        {
            if (!m_Header.Filter.Equals(filter))
            {
                m_Header.Version++;
                m_Header.Filter = filter;
            }
        }

        public void BakeTransform(AffineTransform transform)
        {
            SafetyChecks.LogWarning($"Not implemented for collider type {Type}.");
        }

        /// <summary>
        /// Indicates whether collider should collide normally with others,
        /// or skip collision, but still move and intercept queries.
        /// </summary>
        internal bool RespondsToCollision => Material.CollisionResponse != CollisionResponsePolicy.None;

        /// <summary>
        /// Sets a material field for this voxel collider.
        /// </summary>
        internal void SetMaterialField(Material material, Material.MaterialField option)
        {
            if (option == Material.MaterialField.Friction || option == Material.MaterialField.FrictionCombinePolicy)
                Material.Friction = material.Friction;
            if (option == Material.MaterialField.Restitution || option == Material.MaterialField.RestitutionCombinePolicy)
                Material.Restitution = material.Restitution;
            if (option == Material.MaterialField.CollisionResponsePolicy)
                Material.CollisionResponse = material.CollisionResponse;
            // Increment version when material changes
            m_Header.Version++;
        }

        /// <summary>
        /// Creates a VoxelCollider from a sector map.
        /// Note: Mass properties are initialized with default values (mass=1, inertia tensor identity).
        /// Call SetMassProperties() on the created collider to apply computed mass properties.
        /// </summary>
        public static BlobAssetReference<Collider> Create(
            IDictionary<Vector3Int, SectorHandle> sectorMap,
            CollisionFilter filter, Material material)
        {
            unsafe
            {
                var collider = default(VoxelCollider);

                // Initialize header
                collider.m_Header.Type = ColliderType.Voxel;
                collider.m_Header.CollisionType = CollisionType.Voxel;
                collider.m_Header.Version = 0;
                collider.m_Header.Magic = 0xff;
                collider.m_Header.ForceUniqueBlobID = ~ColliderConstants.k_SharedBlobID;
                collider.m_Header.Filter = filter;

                // Initialize material
                collider.Material = material;

                // Initialize with default mass properties
                // These should be updated using SetMassProperties() after computing actual values
                collider.m_MassProperties = new MassProperties
                {
                    MassDistribution = new MassDistribution
                    {
                        Transform = new RigidTransform(quaternion.identity, float3.zero),
                        InertiaTensor = new float3(1.0f, 1.0f, 1.0f) // Default identity-like inertia
                    },
                    Volume = 1.0f,
                    AngularExpansionFactor = 0.0f
                };

                // Initialize sectors
                collider.m_Sectors = new UnsafeHashMap<int3, SectorHandle>(sectorMap?.Count ?? 1, Allocator.Persistent);
                collider.ReloadSectors(sectorMap);

                return BlobAssetReference<Collider>.Create(&collider, sizeof(VoxelCollider));
            }
        }

        public void ReloadSectors(IDictionary<Vector3Int, SectorHandle> sectorMap)
        {
            if (sectorMap == null) return;
            m_Sectors.Clear();
            foreach (var kvp in sectorMap)
            {
                m_Sectors.Add(
                    new int3(kvp.Key.x, kvp.Key.y, kvp.Key.z), kvp.Value);
            }
        }

        public void Dispose()
        {
            m_Sectors.Dispose();
        }
    }
}