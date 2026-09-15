using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Caelix.Simulation;

namespace Caelix
{
    public partial struct VoxelBodyData : IDisposable
    {
        public struct MassProperties
        {
            public float mass;
            public float3 centerOfMass;
            public float3 inertiaTensor;
        }

        private Allocator allocator;
        private UnsafeHashMap<int3, VoxelEntityPhysics.MassMoments> regionMassCache;
        private VoxelEntityPhysics.MassMoments cachedMassMoments;
        private bool massCacheInitialized;

        /// <summary>
        /// Selects the Unity Physics solver used for
        /// contacts involving this body: <see cref="SolverType.Direct"/> when true, otherwise
        /// <see cref="SolverType.Iterative"/>. Authored on the body component and applied when the body is added.
        /// </summary>
        public bool accuratePhysics;

        public BlobAssetReference<Collider> collider;
        public Unity.Physics.MotionData motionData;
        public Unity.Physics.MotionVelocity motionVelocity;
        public MassProperties massProperties { get; private set; }

        public int _cached_body_index;

        public VoxelBodyData(Allocator allocator)
        {
            this.allocator = allocator;
            regionMassCache = default;
            cachedMassMoments = default;
            massCacheInitialized = false;
            accuratePhysics = true;
            collider = default;
            motionData = DefaultMotionData();
            motionVelocity = DefaultMotionVelocity();
            massProperties = default;
            _cached_body_index = -1;
        }

        /// <summary>
        /// Creates a body record with its voxel collider. The owning world calls this when a body
        /// is added and disposes the record when the body is removed.
        /// </summary>
        public static VoxelBodyData Create(Allocator allocator, bool accuratePhysics)
        {
            var data = new VoxelBodyData(allocator);
            data.accuratePhysics = accuratePhysics;

            var material = new Unity.Physics.Material
            {
                Friction = 0.04f,
                Restitution = 0.0f,
                FrictionCombinePolicy = Unity.Physics.Material.CombinePolicy.GeometricMean,
                RestitutionCombinePolicy = Unity.Physics.Material.CombinePolicy.GeometricMean,
                CollisionResponse = Unity.Physics.CollisionResponsePolicy.CollideRaiseCollisionEvents
            };

            data.collider = Unity.Physics.VoxelCollider.Create(
                default,
                Unity.Physics.CollisionFilter.Default,
                material);
            return data;
        }

        private static Unity.Physics.MotionData DefaultMotionData()
        {
            return new Unity.Physics.MotionData
            {
                WorldFromMotion = RigidTransform.identity,
                BodyFromMotion = RigidTransform.identity,
                LinearDamping = 0.01f,
                AngularDamping = 0.05f
            };
        }

        private static Unity.Physics.MotionVelocity DefaultMotionVelocity()
        {
            return new Unity.Physics.MotionVelocity
            {
                LinearVelocity = float3.zero,
                AngularVelocity = float3.zero,
                InverseInertia = float3.zero,
                InverseMass = 0.0f,
                AngularExpansionFactor = 0.0f,
                GravityFactor = 1.0f
            };
        }

        /// <summary>
        /// Refreshes mass properties and the physics slot from the owning entity's voxel data.
        /// Called by the server world after dirty propagation and before forces and simulation.
        /// </summary>
        public MassProperties ComputePhysicsProperties(in VoxelEntityData entity)
        {
            RefreshMassPropertiesCache(in entity, entity.isStatic);
            RefreshPhysicsSlot(in entity);
            return massProperties;
        }

        private void RefreshMassPropertiesCache(
            in VoxelEntityData entity,
            bool isStatic,
            BrickUpdateFlags dirtyMask = BrickUpdateFlags.Geometry)
        {
            if (isStatic)
            {
                ClearMassPropertiesCache();
                massProperties = default;
                return;
            }

            int regionCount = entity.RegionCount;
            if (regionCount == 0)
            {
                ClearMassPropertiesCache();
                massProperties = default;
                return;
            }

            bool resetCache = !massCacheInitialized || !regionMassCache.IsCreated;
            EnsureMassPropertiesCache(regionCount, resetCache);

            bool changed = RemoveMissingRegionMoments(in entity);

            var inputs = new NativeList<VoxelEntityPhysics.RegionMassMomentInput>(Allocator.TempJob);
            var required = new NativeList<RequiredBrick>(Allocator.TempJob);
            var selected = new NativeHashSet<int3>(math.max(1, regionCount), Allocator.Temp);
            var regions = entity.GetRegionPositions(Allocator.Temp);
            try
            {
                // A region with no cached moments has never been summed, so it is always work.
                for (int i = 0; i < regions.Length; i++)
                {
                    if (!regionMassCache.ContainsKey(regions[i]))
                    {
                        selected.Add(regions[i]);
                    }
                }

                // Otherwise consume the work selected by server dirty propagation, just as
                // RefreshPhysicsSlot does. Source dirty flags remain alive for replication
                // until EndTick; they describe writes, rather than scheduled consumer work.
                entity.CollectRequiredBricks(default, dirtyMask, false, required);
                for (int i = 0; i < required.Length; i++)
                {
                    selected.Add(VoxelRegion.OfKey(required[i].Key));
                }

                foreach (int3 regionPos in selected)
                {
                    inputs.Add(new VoxelEntityPhysics.RegionMassMomentInput
                    {
                        RegionPos = regionPos,
                        Entity = entity
                    });
                }

                if (inputs.Length == 0)
                {
                    if (changed)
                    {
                        ApplyCachedMassProperties();
                    }
                    return;
                }

                using var results = new NativeArray<VoxelEntityPhysics.RegionMassMomentResult>(inputs.Length, Allocator.TempJob);
                var job = new VoxelEntityPhysics.ComputeRegionMassMomentsJob
                {
                    settings = PhysicsSettings.Settings,
                    inputs = inputs.AsArray(),
                    results = results
                };
                job.Schedule(inputs.Length, 1).Complete();

                for (int i = 0; i < results.Length; i++)
                {
                    VoxelEntityPhysics.RegionMassMomentResult result = results[i];
                    VoxelEntityPhysics.MassMoments oldMoments = default;
                    bool hadCachedRegion = regionMassCache.TryGetValue(result.RegionPos, out oldMoments);

                    if (hadCachedRegion)
                    {
                        regionMassCache[result.RegionPos] = result.Moments;
                    }
                    else
                    {
                        regionMassCache.Add(result.RegionPos, result.Moments);
                    }

                    cachedMassMoments += result.Moments - oldMoments;
                }

                ApplyCachedMassProperties();
            }
            finally
            {
                if (regions.IsCreated)
                {
                    regions.Dispose();
                }

                if (selected.IsCreated)
                {
                    selected.Dispose();
                }

                if (required.IsCreated)
                {
                    required.Dispose();
                }

                if (inputs.IsCreated)
                {
                    inputs.Dispose();
                }
            }
        }

        private void ClearMassPropertiesCache()
        {
            if (regionMassCache.IsCreated)
            {
                regionMassCache.Dispose();
            }

            regionMassCache = default;
            cachedMassMoments = default;
            massCacheInitialized = false;
        }

        public void Dispose()
        {
            ClearMassPropertiesCache();

            if (collider.IsCreated)
            {
                unsafe
                {
                    if (collider.Value.Type == ColliderType.Voxel)
                    {
                        var vc = (VoxelCollider*)collider.GetUnsafePtr();
                        vc->Dispose();
                    }
                }
                collider.Dispose();
            }
        }

        private void EnsureMassPropertiesCache(int regionCount, bool rebuild)
        {
            if (rebuild)
            {
                ClearMassPropertiesCache();
            }

            if (!regionMassCache.IsCreated)
            {
                regionMassCache = new UnsafeHashMap<int3, VoxelEntityPhysics.MassMoments>(
                    math.max(1, regionCount),
                    allocator == Allocator.Invalid ? Allocator.Persistent : allocator);
            }
            else if (regionMassCache.Capacity < regionCount)
            {
                regionMassCache.Capacity = regionCount;
            }

            massCacheInitialized = true;
        }

        private bool RemoveMissingRegionMoments(in VoxelEntityData entity)
        {
            if (!regionMassCache.IsCreated || regionMassCache.Count == 0)
            {
                return false;
            }

            bool changed = false;
            using var cachedKeys = regionMassCache.GetKeyArray(Allocator.Temp);
            for (int i = 0; i < cachedKeys.Length; i++)
            {
                int3 regionPos = cachedKeys[i];
                if (entity.HasRegion(regionPos))
                {
                    continue;
                }

                cachedMassMoments -= regionMassCache[regionPos];
                regionMassCache.Remove(regionPos);
                changed = true;
            }

            return changed;
        }

        private void ApplyCachedMassProperties()
        {
            MassProperties result = default;
            result.mass = cachedMassMoments.Mass;
            if (cachedMassMoments.Mass > 0f)
            {
                result.centerOfMass = cachedMassMoments.FirstMoment / cachedMassMoments.Mass;
                result.inertiaTensor = VoxelEntityPhysics.InertiaAroundCenterOfMass(cachedMassMoments, result.centerOfMass);
            }

            massProperties = result;
        }
    }
}
