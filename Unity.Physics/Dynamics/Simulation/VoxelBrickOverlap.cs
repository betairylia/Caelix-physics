using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Voxelis;
using static Unity.Physics.Math;

namespace Unity.Physics
{
    /// <summary>
    /// One raw overlap between a queried brick and an allocated brick of another voxel body.
    /// Coordinates are global brick coordinates in each collider's local voxel grid, including
    /// sector offsets. Results are unsorted and may repeat when the input contains duplicate
    /// queries or both endpoints of the same pair are queried.
    /// </summary>
    public struct VoxelBrickOverlapCandidate
    {
        public int BodyIndexA;
        public int BodyIndexB;
        public int3 BrickCoordsInA;
        public int3 BrickCoordsInB;
    }

    /// <summary>Post-simulation voxel-brick queries over a collision world's BVH4 broadphase.</summary>
    public static class VoxelBrickOverlapQueryExtensions
    {
        // REVIEW: VIBE: Why queryBricksByBody is a ParallelMultiHashMap? Seems a NativeStream is more fit.
        // #low-performant
        /// <summary>
        /// Schedules one BVH traversal per queried source body, then tests all of that body's
        /// queried bricks against each candidate target body. The input maps transient collision-
        /// world body indices to global brick coordinates in that body's local voxel grid.
        /// </summary>
        /// <remarks>
        /// <paramref name="queryBricksByBody"/> is suitable for parallel construction through
        /// <see cref="NativeParallelMultiHashMap{TKey,TValue}.ParallelWriter"/>. Pass the producer
        /// handle in <paramref name="inputDeps"/>; no main-thread completion is required.
        ///
        /// The collision world's dynamic tree must describe the desired pose. When this runs
        /// after a simulation step, set <see cref="SimulationStepInput.SynchronizeCollisionWorld"/>
        /// so the solver-integrated transforms and dynamic BVH are synchronized first.
        ///
        /// The returned stream uses <see cref="Allocator.TempJob"/> and belongs to the caller.
        /// Dispose it after its consumers complete. A stream for-each index is the source body
        /// index; raw candidates retain transient body indices for the voxel-side graph builder.
        /// </remarks>
        public static JobHandle ScheduleVoxelBrickOverlaps(
            this CollisionWorld collisionWorld,
            NativeParallelMultiHashMap<int, int3> queryBricksByBody,
            out NativeStream overlaps,
            JobHandle inputDeps = default)
        {
            // NativeStream requires at least one lane. Worlds with no bodies schedule no query
            // work and leave that sole lane empty.
            int laneCount = math.max(1, collisionWorld.NumBodies);
            overlaps = new NativeStream(laneCount, Allocator.TempJob);

            if (collisionWorld.NumBodies == 0)
            {
                return inputDeps;
            }

            return new FindVoxelBrickOverlapsJob
            {
                CollisionWorld = collisionWorld,
                QueryBricksByBody = queryBricksByBody,
                OverlapWriter = overlaps.AsWriter()
            }.Schedule(collisionWorld.NumBodies, 1, inputDeps);
        }
    }

    [BurstCompile]
    internal unsafe struct FindVoxelBrickOverlapsJob : IJobParallelFor
    {
        // Alien-aware voxel reads use a Moore radius of one voxel. Keeping this compile-time
        // policy here makes the query independent of the regular physics contact tolerance.
        const float k_QueryHaloInVoxels = 1.0f;

        [ReadOnly] public CollisionWorld CollisionWorld;
        [ReadOnly] public NativeParallelMultiHashMap<int, int3> QueryBricksByBody;
        public NativeStream.Writer OverlapWriter;

        public void Execute(int sourceBodyIndex)
        {
            OverlapWriter.BeginForEachIndex(sourceBodyIndex);
            QueryBody(sourceBodyIndex, ref OverlapWriter);
            OverlapWriter.EndForEachIndex();
        }

        // REVIEW: VIBE: Remove safety checks / allocated checks here since it can be guranteed by upstream.
        // or at least gate it behind some compiler flags.
        // #over-defensive
        void QueryBody(int sourceBodyIndex, ref NativeStream.Writer writer)
        {
            if (!QueryBricksByBody.TryGetFirstValue(
                sourceBodyIndex, out int3 sourceBrick,
                out NativeParallelMultiHashMapIterator<int> iterator))
            {
                return;
            }

            NativeArray<RigidBody> bodies = CollisionWorld.Bodies;
            RigidBody sourceBody = bodies[sourceBodyIndex];
            Collider* sourceCollider = (Collider*)sourceBody.Collider.GetUnsafePtr();
            if (sourceCollider == null || sourceCollider->CollisionType != CollisionType.Voxel)
            {
                return;
            }

            VoxelCollider* sourceVoxel = (VoxelCollider*)sourceCollider;
            var validSourceBricks = new NativeList<int3>(16, Allocator.Temp);
            Aabb aggregateQueryAabb = Aabb.Empty;

            do
            {
                if (!IsAllocatedBrick(sourceVoxel, sourceBrick))
                {
                    continue;
                }

                validSourceBricks.Add(sourceBrick);
                aggregateQueryAabb.Include(QueryWorldAabb(sourceBody, sourceBrick));
            }
            while (QueryBricksByBody.TryGetNextValue(out sourceBrick, ref iterator));

            if (validSourceBricks.IsEmpty)
            {
                validSourceBricks.Dispose();
                return;
            }

            // CollisionWorld.OverlapAabb traverses the 4-ary broadphase and tests the query
            // against four child bounds at a time through FourTransposedAabbs.
            // If entities routinely submit far-apart query bricks, spatially cluster this list
            // and issue one aggregate traversal per cluster to avoid filling the gaps in this AABB.
            var bodyHits = new NativeList<int>(16, Allocator.Temp);
            CollisionWorld.OverlapAabb(new OverlapAabbInput
            {
                Aabb = aggregateQueryAabb,
                Filter = sourceCollider->GetCollisionFilter()
            }, ref bodyHits);

            NativeArray<int3> sourceBricks = validSourceBricks.AsArray();
            for (int i = 0; i < bodyHits.Length; i++)
            {
                int targetBodyIndex = bodyHits[i];
                RigidBody targetBody = bodies[targetBodyIndex];
                Collider* targetCollider = (Collider*)targetBody.Collider.GetUnsafePtr();

                // TODO: Revise if this check is necessary
                if (!ShouldIncludeTargetBody(
                    sourceBodyIndex, sourceCollider, targetBodyIndex, targetCollider))
                {
                    continue;
                }

                EmitTargetBricks(
                    sourceBodyIndex, sourceBricks, sourceBody,
                    targetBodyIndex, targetBody, (VoxelCollider*)targetCollider, ref writer);
            }

            bodyHits.Dispose();
            validSourceBricks.Dispose();
        }

        static Aabb QueryWorldAabb(in RigidBody sourceBody, int3 sourceBrick)
        {
            float3 localMin = (float3)(sourceBrick * Sector.SIZE_IN_BLOCKS) - k_QueryHaloInVoxels;
            float3 localMax = (float3)((sourceBrick + 1) * Sector.SIZE_IN_BLOCKS) + k_QueryHaloInVoxels;
            return TransformAabb(
                new Aabb { Min = localMin, Max = localMax },
                sourceBody.WorldFromBody);
        }

        static bool ShouldIncludeTargetBody(
            int sourceBodyIndex,
            Collider* sourceCollider,
            int targetBodyIndex,
            Collider* targetCollider)
        {
            if (targetBodyIndex == sourceBodyIndex || targetCollider == null ||
                targetCollider->CollisionType != CollisionType.Voxel)
            {
                return false;
            }

            // Keep the old graph's physics-approved body policy. This defensive filter check is
            // intentionally local even though the BVH leaf processor already applies it.
            if (!sourceCollider->RespondsToCollision || !targetCollider->RespondsToCollision ||
                !CollisionFilter.IsCollisionEnabled(
                    sourceCollider->GetCollisionFilter(), targetCollider->GetCollisionFilter()))
            {
                return false;
            }

            VoxelCollider* sourceVoxel = (VoxelCollider*)sourceCollider;
            VoxelCollider* targetVoxel = (VoxelCollider*)targetCollider;
            return sourceVoxel->m_Sectors.IsCreated && !sourceVoxel->m_Sectors.IsEmpty &&
                targetVoxel->m_Sectors.IsCreated && !targetVoxel->m_Sectors.IsEmpty &&
                Material.GetCombinedCollisionResponse(sourceVoxel->Material, targetVoxel->Material) !=
                CollisionResponsePolicy.None;
        }

        static void EmitTargetBricks(
            int sourceBodyIndex,
            NativeArray<int3> sourceBricks,
            in RigidBody sourceBody,
            int targetBodyIndex,
            in RigidBody targetBody,
            VoxelCollider* targetVoxel,
            ref NativeStream.Writer writer)
        {
            MTransform worldFromSource = new MTransform(sourceBody.WorldFromBody);
            MTransform worldFromTarget = new MTransform(targetBody.WorldFromBody);
            MTransform targetFromSource = Mul(Inverse(worldFromTarget), worldFromSource);

            float3x3 rotation = targetFromSource.Rotation;
            float3 rowAbsSum = math.abs(rotation.c0) + math.abs(rotation.c1) + math.abs(rotation.c2);

            for (int sourceIndex = 0; sourceIndex < sourceBricks.Length; sourceIndex++)
            {
                int3 sourceBrick = sourceBricks[sourceIndex];
                // A one-voxel dilation reproduces the previous graph's alien-neighborhood
                // reach without changing regular broadphase or narrowphase tolerances.
                ManifoldQueries.GetOverlappingBrickRange(
                    sourceBrick * Sector.SIZE_IN_BLOCKS,
                    targetFromSource,
                    rowAbsSum,
                    k_QueryHaloInVoxels,
                    out int3 brickLo,
                    out int3 brickHi);

                EmitAllocatedTargetRange(
                    sourceBodyIndex, sourceBrick, targetBodyIndex,
                    brickLo, brickHi, targetVoxel, ref writer);
            }
        }

        static void EmitAllocatedTargetRange(
            int sourceBodyIndex,
            int3 sourceBrick,
            int targetBodyIndex,
            int3 brickLo,
            int3 brickHi,
            VoxelCollider* targetVoxel,
            ref NativeStream.Writer writer)
        {
            int3 cachedSectorCoord = default;
            SectorHandle cachedSector = default;
            bool cacheValid = false;
            bool cachedSectorExists = false;

            for (int z = brickLo.z; z <= brickHi.z; z++)
            {
                for (int y = brickLo.y; y <= brickHi.y; y++)
                {
                    for (int x = brickLo.x; x <= brickHi.x; x++)
                    {
                        int3 targetBrick = new int3(x, y, z);
                        int3 sectorCoord = targetBrick >> Sector.SHIFT_IN_BRICKS;
                        if (!cacheValid || math.any(sectorCoord != cachedSectorCoord))
                        {
                            cachedSectorExists = targetVoxel->m_Sectors.TryGetValue(
                                sectorCoord, out cachedSector) && !cachedSector.IsNull;
                            cachedSectorCoord = sectorCoord;
                            cacheValid = true;
                        }
                        if (!cachedSectorExists)
                        {
                            continue;
                        }

                        int3 brickInSector = targetBrick & Sector.SECTOR_MASK;
                        short brickId = cachedSector.Ptr->brickIdx[Sector.ToBrickIdx(
                            brickInSector.x, brickInSector.y, brickInSector.z)];
                        if (brickId == Sector.BRICKID_EMPTY ||
                            !ShouldIncludeTargetBrick(
                                targetVoxel, targetBrick, cachedSector, brickId))
                        {
                            continue;
                        }

                        writer.Write(new VoxelBrickOverlapCandidate
                        {
                            BodyIndexA = sourceBodyIndex,
                            BodyIndexB = targetBodyIndex,
                            BrickCoordsInA = sourceBrick,
                            BrickCoordsInB = targetBrick
                        });
                    }
                }
            }
        }

        // TODO: This is not source-aware.
        /// <summary>
        /// Compile-time target-brick policy hook. Add future persistent-flag or voxel-property
        /// filtering here; the public API intentionally has no runtime policy parameters yet.
        /// </summary>
        static bool ShouldIncludeTargetBrick(
            VoxelCollider* targetVoxel,
            int3 targetBrick,
            SectorHandle targetSector,
            short targetBrickId)
        {
            return true;
        }

        static bool IsAllocatedBrick(VoxelCollider* voxel, int3 brickCoord)
        {
            if (!voxel->m_Sectors.IsCreated || voxel->m_Sectors.IsEmpty)
            {
                return false;
            }

            int3 sectorCoord = brickCoord >> Sector.SHIFT_IN_BRICKS;
            if (!voxel->m_Sectors.TryGetValue(sectorCoord, out SectorHandle sector) || sector.IsNull)
            {
                return false;
            }

            int3 brickInSector = brickCoord & Sector.SECTOR_MASK;
            return sector.Ptr->brickIdx[Sector.ToBrickIdx(
                brickInSector.x, brickInSector.y, brickInSector.z)] != Sector.BRICKID_EMPTY;
        }
    }
}
