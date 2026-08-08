using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Physics.Math;

namespace Unity.Physics
{
    /// <summary>
    /// One raw brick-overlap candidate emitted by the voxel narrowphase marking pass.
    /// A candidate means the allocated bricks passed <c>MarkOverlappedBricksInB</c> with its
    /// configured contact/halo window, not that individual occupied blocks intersect.
    ///
    /// Brick coordinates are global brick coordinates in each collider's local voxel grid,
    /// including sector offsets. Raw candidates are unsorted, but the producer emits each
    /// physical brick pair exactly once. Body indices are only valid for the step that produced
    /// them.
    /// </summary>
    public struct VoxelBrickOverlapCandidate
    {
        public int BodyIndexA;
        public int BodyIndexB;
        public int3 BrickCoordsInA;
        public int3 BrickCoordsInB;
    }

    /// <summary>
    /// Emission target handed down into the voxel brick marking pass. A null writer disables
    /// emission. BodyIndices always hold the original (unswapped) physics pair; Flipped is set
    /// when the marking runs with A/B roles swapped, so the emitted coordinates keep their
    /// original body correspondence.
    /// </summary>
    internal unsafe struct VoxelBrickOverlapEmission
    {
        public NativeStream.Writer* Writer;
        public BodyIndexPair BodyIndices;
        public bool Flipped;
    }

    /// <summary>
    /// Burst-compatible view over the raw dynamic-pair and static-static brick-overlap
    /// candidate streams of one step.
    ///
    /// The view is valid after the step's FinalExecutionHandle completes and until the next
    /// simulation reset or disposal. Either stream may be uncreated (for example the dynamic
    /// stream in a world with zero dynamic bodies); <see cref="Count"/> treats it as empty.
    /// </summary>
    public struct VoxelBrickOverlapCandidates
    {
        [NativeDisableContainerSafetyRestriction]
        readonly NativeStream m_DynamicStream;
        [NativeDisableContainerSafetyRestriction]
        readonly NativeStream m_StaticStream;

        public VoxelBrickOverlapCandidates(NativeStream dynamicStream, NativeStream staticStream)
        {
            m_DynamicStream = dynamicStream;
            m_StaticStream = staticStream;
        }

        /// <summary> Raw dynamic-pair candidate stream. May be uncreated. For advanced (parallel) consumption. </summary>
        public NativeStream DynamicStream => m_DynamicStream;

        /// <summary> Raw static-static candidate stream. May be uncreated. For advanced (parallel) consumption. </summary>
        public NativeStream StaticStream => m_StaticStream;

        /// <summary> Total number of raw candidates in both streams. </summary>
        public int Count()
        {
            return (m_DynamicStream.IsCreated ? m_DynamicStream.Count() : 0)
                + (m_StaticStream.IsCreated ? m_StaticStream.Count() : 0);
        }
    }

    /// <summary>
    /// Consumes the static-static broadphase body-pair stream and emits brick-overlap
    /// candidates for voxel-voxel pairs. Marking only: no key-block traversal, no contacts,
    /// no manifolds, no solver work. Pairs already passed the broadphase collision filter
    /// and RespondsToCollision checks; combined-material CollisionResponse.None pairs and
    /// non-voxel pairs are skipped here.
    /// </summary>
    [BurstCompile]
    internal unsafe struct CreateStaticVoxelBrickOverlapCandidatesJob : IJob
    {
        [ReadOnly] public NativeArray<RigidBody> Bodies;
        [ReadOnly] public NativeStream PairStream;
        public float CollisionTolerance;
        public NativeStream.Writer CandidateWriter;

        public void Execute()
        {
            CandidateWriter.BeginForEachIndex(0);

            var reader = PairStream.AsReader();
            int forEachCount = PairStream.ForEachCount;
            for (int i = 0; i < forEachCount; i++)
            {
                reader.BeginForEachIndex(i);
                while (reader.RemainingItemCount > 0)
                {
                    var overlap = reader.Read<Broadphase.OverlapResult>();
                    ProcessPair(overlap.BodyPair);
                }
                reader.EndForEachIndex();
            }

            CandidateWriter.EndForEachIndex();
        }

        void ProcessPair(BodyIndexPair pair)
        {
            RigidBody rigidBodyA = Bodies[pair.BodyIndexA];
            RigidBody rigidBodyB = Bodies[pair.BodyIndexB];

            Collider* colliderA = (Collider*)rigidBodyA.Collider.GetUnsafePtr();
            Collider* colliderB = (Collider*)rigidBodyB.Collider.GetUnsafePtr();
            if (colliderA == null || colliderB == null)
            {
                return;
            }

            // Defensive re-check; the broadphase pair writer already filters.
            if (!CollisionFilter.IsCollisionEnabled(colliderA->GetCollisionFilter(), colliderB->GetCollisionFilter()))
            {
                return;
            }

            if (colliderA->CollisionType != CollisionType.Voxel || colliderB->CollisionType != CollisionType.Voxel)
            {
                return;
            }

            VoxelCollider* voxelA = (VoxelCollider*)colliderA;
            VoxelCollider* voxelB = (VoxelCollider*)colliderB;

            if (Material.GetCombinedCollisionResponse(voxelA->Material, voxelB->Material)
                == CollisionResponsePolicy.None)
            {
                return;
            }

            if (!voxelA->m_Sectors.IsCreated || !voxelB->m_Sectors.IsCreated ||
                voxelA->m_Sectors.IsEmpty || voxelB->m_Sectors.IsEmpty)
            {
                return;
            }

            var worldFromA = new MTransform(rigidBodyA.WorldFromBody);
            var worldFromB = new MTransform(rigidBodyB.WorldFromBody);

            var writer = (NativeStream.Writer*)UnsafeUtility.AddressOf(ref CandidateWriter);

            // Same size-based role swap as the dynamic VoxelVoxel path, so both paths walk
            // the smaller body's bricks. Flipped keeps the emitted coordinates on the
            // original bodies.
            // TODO: Warp this so a single source-of-truth for cost estimation.
            bool swap = voxelA->m_Sectors.Count > voxelB->m_Sectors.Count;
            if (swap)
            {
                ManifoldQueries.MarkVoxelBrickOverlapCandidates(
                    voxelB, voxelA, Mul(Inverse(worldFromA), worldFromB), CollisionTolerance,
                    new VoxelBrickOverlapEmission { Writer = writer, BodyIndices = pair, Flipped = true });
            }
            else
            {
                ManifoldQueries.MarkVoxelBrickOverlapCandidates(
                    voxelA, voxelB, Mul(Inverse(worldFromB), worldFromA), CollisionTolerance,
                    new VoxelBrickOverlapEmission { Writer = writer, BodyIndices = pair, Flipped = false });
            }
        }
    }
}
