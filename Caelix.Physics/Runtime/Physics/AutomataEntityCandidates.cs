using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Profiling;
using Caelix.Utils;

namespace Caelix.Simulation
{
    public sealed partial class VoxelPhysicsWorld
    {
        private static readonly ProfilerMarker AutomataCandidatesMarker = new("Automata Entity Candidates");
        private NativeParallelHashMap<Guid128, int> alienEntityRanks;

        /// <summary>
        /// Queries the prepared physics BVH once per source entity, then narrows those entity
        /// candidates for each required brick. Results are indices into the supplied view order,
        /// preserving occupancy priority independently of BVH traversal or body-index order.
        /// No target bricks are enumerated. The caller owns and reuses the result buffer.
        /// </summary>
        public void CollectAutomataCandidates(NativeList<RequiredBrick> bricks,
            NativeArray<AlienEntityView> entities, NativeList<int> candidates)
        {
            using var marker = AutomataCandidatesMarker.Auto();
            if (!spatialPrepared) throw new System.InvalidOperationException("Prepare spatial queries before collecting candidates.");
            if (!alienEntityRanks.IsCreated)
                alienEntityRanks = new NativeParallelHashMap<Guid128, int>(math.max(1, entities.Length), Allocator.Persistent);
            alienEntityRanks.Clear();
            if (alienEntityRanks.Capacity < entities.Length) alienEntityRanks.Capacity = entities.Length;
            for (int i = 0; i < entities.Length; i++) alienEntityRanks.Add(entities[i].EntityId, i);

            new CollectAutomataCandidatesJob
            {
                World = physicsWorld.CollisionWorld,
                BodyGuids = bodyIndexToGuid,
                EntityRanks = alienEntityRanks,
                Entities = entities,
                Bricks = bricks,
                Candidates = candidates
            }.Schedule().Complete();
        }

        [BurstCompile]
        private struct CollectAutomataCandidatesJob : IJob
        {
            [ReadOnly] public CollisionWorld World;
            [ReadOnly] public NativeArray<Guid128> BodyGuids;
            [ReadOnly] public NativeParallelHashMap<Guid128, int> EntityRanks;
            [ReadOnly] public NativeArray<AlienEntityView> Entities;
            public NativeList<RequiredBrick> Bricks;
            public NativeList<int> Candidates;

            public void Execute()
            {
                Candidates.Clear();
                var hits = new NativeList<int>(16, Allocator.Temp);
                var ranks = new NativeList<int>(16, Allocator.Temp);
                // Collection appends each entity's work contiguously. Multiple runs of the same
                // entity are also valid; they simply get separate aggregate queries.
                for (int first = 0; first < Bricks.Length;)
                {
                    Guid128 source = Bricks[first].Entity;
                    int end = first;
                    Aabb aggregate = Aabb.Empty;
                    while (end < Bricks.Length && Bricks[end].Entity == source)
                        aggregate.Include(QueryBounds(Bricks[end++]));

                    hits.Clear();
                    ranks.Clear();
                    // This is the same BVH traversal used by the post-step brick query. Stop at
                    // entities here: repeated voxel reads bind target storage through BrickCursor.
                    World.OverlapAabb(new OverlapAabbInput
                    {
                        Aabb = aggregate, Filter = CollisionFilter.Default
                    }, ref hits);
                    for (int h = 0; h < hits.Length; h++)
                    {
                        int body = hits[h];
                        if ((uint)body < (uint)BodyGuids.Length && BodyGuids[body] != source &&
                            EntityRanks.TryGetValue(BodyGuids[body], out int rank)) ranks.Add(rank);
                    }
                    ranks.AsArray().Sort();

                    for (int b = first; b < end; b++)
                    {
                        RequiredBrick brick = Bricks[b];
                        Aabb bounds = QueryBounds(brick);
                        int start = Candidates.Length;
                        int previous = -1;
                        for (int r = 0; r < ranks.Length; r++)
                        {
                            int rank = ranks[r];
                            if (rank == previous) continue;
                            previous = rank;
                            AlienEntityView target = Entities[rank];
                            if (math.all(bounds.Max >= target.WorldAabbMin) && math.all(bounds.Min <= target.WorldAabbMax))
                                Candidates.Add(rank);
                        }
                        brick.AlienCandidateRange = new int2(start, Candidates.Length - start);
                        Bricks[b] = brick;
                    }
                    first = end;
                }
                ranks.Dispose();
                hits.Dispose();
            }

            private static Aabb QueryBounds(in RequiredBrick brick)
            {
                float3 origin = BrickKey.ToBlockOrigin(brick.Key);
                float3 center = math.transform(brick.LocalToWorld, origin + BrickKey.BlocksPerAxis * 0.5f);
                float3x3 rotation = new float3x3(brick.LocalToWorld.rot);
                float3 extent = (math.abs(rotation.c0) + math.abs(rotation.c1) + math.abs(rotation.c2))
                    * (BrickKey.BlocksPerAxis * 0.5f + 1f);
                return new Aabb { Min = center - extent, Max = center + extent };
            }
        }
    }
}
