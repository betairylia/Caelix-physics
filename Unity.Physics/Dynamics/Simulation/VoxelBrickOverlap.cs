using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Voxelis;
using static Unity.Physics.Math;

namespace Unity.Physics
{
    /// <summary>
    /// Describes one lane in a voxel-brick query stream.
    /// </summary>
    /// <remarks>
    /// A batch may contain any spatial grouping of bricks. All brick records in the corresponding
    /// stream lane must belong to one source entity, identified by <see cref="SourceBodyIndex"/>.
    /// One source entity may be represented by any number of batches.
    /// </remarks>
    public struct VoxelBrickOverlapQueryBatch
    {
        /// <summary>
        /// The transient <see cref="CollisionWorld"/> body index that identifies the source entity.
        /// </summary>
        public int SourceBodyIndex;
    }

    /// <summary>One source brick record in a voxel-brick query batch.</summary>
    /// <remarks>
    /// <see cref="BrickCoord"/> is a global brick coordinate in the source collider's local
    /// voxel grid, including sector offsets. <see cref="Flags"/> is source metadata forwarded
    /// to the compile-time target-brick policy hook.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential, Size = 16)]
    public struct VoxelBrickOverlapQuery
    {
        public int3 BrickCoord;
        public ushort Flags;

        public VoxelBrickOverlapQuery(int3 brickCoord, ushort flags)
        {
            BrickCoord = brickCoord;
            Flags = flags;
        }
    }

    /// <summary>
    /// One raw overlap between a queried brick and an allocated brick of another voxel body.
    /// Coordinates are global brick coordinates in each collider's local voxel grid, including
    /// sector offsets. Results are unsorted and may repeat when the input contains duplicate
    /// queries, overlapping batches, or both endpoints of the same pair are queried.
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
        /// <summary>
        /// Schedules one BVH traversal per query batch, then tests the batch's source bricks
        /// against each candidate target body.
        /// </summary>
        /// <remarks>
        /// <paramref name="queryBatches"/> and <paramref name="queryBricksByBatch"/> describe the
        /// same batches by index. For a nonempty request, the stream must contain exactly
        /// <c>queryBatches.Length</c> lanes. Stream lane <c>i</c> contains
        /// <see cref="VoxelBrickOverlapQuery"/> records belonging to
        /// <c>queryBatches[i].SourceBodyIndex</c>. A batch may use any grouping, but must not mix
        /// source bodies. Separate batches may identify the same source body.
        ///
        /// The input stream may be constructed by jobs. Pass all input producer handles in
        /// <paramref name="inputDeps"/>; no main-thread completion is required.
        ///
        /// The collision world's dynamic tree must describe the desired pose. When this runs
        /// after a simulation step, set <see cref="SimulationStepInput.SynchronizeCollisionWorld"/>
        /// so the solver-integrated transforms and dynamic BVH are synchronized first.
        ///
        /// The returned stream uses <see cref="Allocator.TempJob"/> and belongs to the caller.
        /// Dispose it after its consumers complete. An output stream for-each index is the input
        /// batch index; raw candidates retain transient body indices for the voxel-side consumer.
        /// </remarks>
        public static JobHandle ScheduleVoxelBrickOverlaps(
            this CollisionWorld collisionWorld,
            NativeArray<VoxelBrickOverlapQueryBatch> queryBatches,
            NativeStream queryBricksByBatch,
            out NativeStream overlaps,
            JobHandle inputDeps = default)
        {
            // NativeStream requires at least one lane. An empty request schedules no query work
            // and leaves that sole lane empty.
            int laneCount = math.max(1, queryBatches.Length);
            overlaps = new NativeStream(laneCount, Allocator.TempJob);

            if (collisionWorld.NumBodies == 0 || queryBatches.Length == 0)
            {
                return inputDeps;
            }

            return new FindVoxelBrickOverlapsJob
            {
                CollisionWorld = collisionWorld,
                QueryBatches = queryBatches,
                QueryBricksByBatch = queryBricksByBatch.AsReader(),
                OverlapWriter = overlaps.AsWriter()
            }.Schedule(queryBatches.Length, 1, inputDeps);
        }
    }

    [BurstCompile]
    internal unsafe struct FindVoxelBrickOverlapsJob : IJobParallelFor
    {
        // Alien-aware voxel reads use a Moore radius of one voxel. Keeping this compile-time
        // policy here makes the query independent of the regular physics contact tolerance.
        const float k_QueryHaloInVoxels = 1.0f;

        [ReadOnly] public CollisionWorld CollisionWorld;
        [ReadOnly] public NativeArray<VoxelBrickOverlapQueryBatch> QueryBatches;
        public NativeStream.Reader QueryBricksByBatch;
        public NativeStream.Writer OverlapWriter;

        public void Execute(int batchIndex)
        {
            OverlapWriter.BeginForEachIndex(batchIndex);
            QueryBatch(batchIndex, ref OverlapWriter);
            OverlapWriter.EndForEachIndex();
        }

        void QueryBatch(int batchIndex, ref NativeStream.Writer writer)
        {
            VoxelBrickOverlapQueryBatch sourceBatch = QueryBatches[batchIndex];
            int sourceBodyIndex = sourceBatch.SourceBodyIndex;
            if ((uint)sourceBodyIndex >= (uint)CollisionWorld.NumBodies)
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

            NativeStream.Reader queryReader = QueryBricksByBatch;
            int queryCount = queryReader.BeginForEachIndex(batchIndex);
            if (queryCount == 0)
            {
                queryReader.EndForEachIndex();
                return;
            }

            var sourceQueries = new NativeList<VoxelBrickOverlapQuery>(queryCount, Allocator.Temp);
            Aabb aggregateQueryAabb = Aabb.Empty;

            for (int queryIndex = 0; queryIndex < queryCount; queryIndex++)
            {
                VoxelBrickOverlapQuery sourceQuery = queryReader.Read<VoxelBrickOverlapQuery>();
                sourceQueries.Add(sourceQuery);
                aggregateQueryAabb.Include(QueryWorldAabb(sourceBody, sourceQuery.BrickCoord));
            }
            queryReader.EndForEachIndex();

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

            NativeArray<VoxelBrickOverlapQuery> sourceBricks = sourceQueries.AsArray();
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
                    sourceBatch, sourceBricks, sourceBody,
                    targetBodyIndex, targetBody, (VoxelCollider*)targetCollider, ref writer);
            }

            bodyHits.Dispose();
            sourceQueries.Dispose();
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

            // Require the same body-level collision eligibility used by physics. This defensive
            // filter check is local even though the BVH leaf processor already applies it.
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
            VoxelBrickOverlapQueryBatch sourceBatch,
            NativeArray<VoxelBrickOverlapQuery> sourceBricks,
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
                VoxelBrickOverlapQuery sourceQuery = sourceBricks[sourceIndex];
                int3 sourceBrick = sourceQuery.BrickCoord;
                // A one-voxel dilation supplies alien-neighborhood reach without changing
                // regular broadphase or narrowphase tolerances.
                ManifoldQueries.GetOverlappingBrickRange(
                    sourceBrick * Sector.SIZE_IN_BLOCKS,
                    targetFromSource,
                    rowAbsSum,
                    k_QueryHaloInVoxels,
                    out int3 brickLo,
                    out int3 brickHi);

                EmitAllocatedTargetRange(
                    sourceBatch, sourceQuery, targetBodyIndex,
                    brickLo, brickHi, targetVoxel, ref writer);
            }
        }

        static void EmitAllocatedTargetRange(
            VoxelBrickOverlapQueryBatch sourceBatch,
            VoxelBrickOverlapQuery sourceQuery,
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
                                sourceBatch, sourceQuery,
                                targetVoxel, targetBrick, cachedSector, brickId))
                        {
                            continue;
                        }

                        writer.Write(new VoxelBrickOverlapCandidate
                        {
                            BodyIndexA = sourceBatch.SourceBodyIndex,
                            BodyIndexB = targetBodyIndex,
                            BrickCoordsInA = sourceQuery.BrickCoord,
                            BrickCoordsInB = targetBrick
                        });
                    }
                }
            }
        }

        /// <summary>
        /// Compile-time target-brick policy hook. The source batch supplies the source entity
        /// identifier and the source query supplies its flags. Target persistent-flag or
        /// voxel-property filtering belongs here.
        /// </summary>
        static bool ShouldIncludeTargetBrick(
            VoxelBrickOverlapQueryBatch sourceBatch,
            VoxelBrickOverlapQuery sourceQuery,
            VoxelCollider* targetVoxel,
            int3 targetBrick,
            SectorHandle targetSector,
            short targetBrickId)
        {
            return true;
        }
    }
}
