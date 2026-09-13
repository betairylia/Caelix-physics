using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Caelix;
using Caelix.Simulation;
using Caelix.Utils;
using Caelix.Tests.TestSupport;

namespace Caelix.Tests
{
    public class VoxelEntityPhysicsTests
    {
        [BurstCompile]
        private unsafe struct CountPhysicsKeyBlocksJob : IJob
        {
            [ReadOnly] public VoxelEntityData Entity;
            [WriteOnly] public NativeArray<int> Result;

            public void Execute()
            {
                int count = 0;
                foreach (int3 key in Entity.EnumerateBricks())
                {
                    foreach (BrickSlotEntry<PhysicsInfo> item in
                             Entity.EnumerateBrickBitmask<PhysicsInfo>(SectorSlotId.PhysicsInfo, key))
                    {
                        count++;
                    }
                }

                Result[0] = count;
            }
        }

        [Test]
        public void RegionMassMomentsForSingleBlockUseVoxelCenter()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(int3.zero, new Block(1));
            scope.Data.RefreshNonEmptyMask();

            VoxelEntityPhysics.MassMoments moments =
                VoxelEntityPhysics.ComputeRegionMassMoments(scope.Data, int3.zero, PhysicsSettings.Settings);

            Assert.That(moments.Mass, Is.EqualTo(1f));
            Assert.That(moments.FirstMoment, Is.EqualTo(new float3(0.5f, 0.5f, 0.5f)));
            Assert.That(moments.InertiaOrigin, Is.EqualTo(new float3(0.5f, 0.5f, 0.5f)));
            Assert.That(VoxelEntityPhysics.InertiaAroundCenterOfMass(moments, moments.FirstMoment / moments.Mass),
                Is.EqualTo(float3.zero));
        }

        [Test]
        public void InertiaAroundCenterOfMassUsesParallelAxisTheorem()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(int3.zero, new Block(1));
            scope.Data.SetBlock(new int3(2, 0, 0), new Block(1));
            scope.Data.RefreshNonEmptyMask();

            VoxelEntityPhysics.MassMoments moments =
                VoxelEntityPhysics.ComputeRegionMassMoments(scope.Data, int3.zero, PhysicsSettings.Settings);

            float3 centerOfMass = moments.FirstMoment / moments.Mass;
            float3 inertia = VoxelEntityPhysics.InertiaAroundCenterOfMass(moments, centerOfMass);

            Assert.That(moments.Mass, Is.EqualTo(2f));
            Assert.That(centerOfMass, Is.EqualTo(new float3(1.5f, 0.5f, 0.5f)));
            Assert.That(inertia, Is.EqualTo(new float3(0f, 2f, 2f)));
        }

        [Test]
        public void RegionMassMomentsIncludeRegionBlockPosition()
        {
            using var scope = new EntityDataTestScope();
            // The first block of region (1,0,0), i.e. one region along x from the origin.
            scope.Data.SetBlock(new int3(VoxelRegion.SizeInBlocks, 0, 0), new Block(1));
            scope.Data.RefreshNonEmptyMask();

            VoxelEntityPhysics.MassMoments moments =
                VoxelEntityPhysics.ComputeRegionMassMoments(
                    scope.Data,
                    new int3(1, 0, 0),
                    PhysicsSettings.Settings);

            Assert.That(moments.Mass, Is.EqualTo(1f));
            Assert.That(moments.FirstMoment,
                Is.EqualTo(new float3(VoxelRegion.SizeInBlocks + 0.5f, 0.5f, 0.5f)));
        }

        [Test]
        public void VoxelBodyDataComputesMassPropertiesFromEntityStorage()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(int3.zero, new Block(1));
            scope.Data.RefreshNonEmptyMask();

            var bodyData = new VoxelBodyData(Allocator.Persistent);
            try
            {
                VoxelBodyData.MassProperties massProperties =
                    bodyData.ComputePhysicsProperties(scope.Data);

                Assert.That(massProperties.mass, Is.EqualTo(1f));
                Assert.That(massProperties.centerOfMass, Is.EqualTo(new float3(0.5f, 0.5f, 0.5f)));
                Assert.That(massProperties.inertiaTensor, Is.EqualTo(float3.zero));
            }
            finally
            {
                bodyData.Dispose();
            }
        }

        [Test]
        public void VoxelBodyDataClearsMassPropertiesForStaticBodies()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(int3.zero, new Block(1));
            scope.Data.RefreshNonEmptyMask();

            var bodyData = new VoxelBodyData(Allocator.Persistent);
            try
            {
                Assert.That(bodyData.ComputePhysicsProperties(scope.Data).mass, Is.EqualTo(1f));

                scope.Data.isStatic = true;
                VoxelBodyData.MassProperties massProperties =
                    bodyData.ComputePhysicsProperties(scope.Data);

                Assert.That(massProperties.mass, Is.EqualTo(0f));
                Assert.That(massProperties.centerOfMass, Is.EqualTo(float3.zero));
                Assert.That(massProperties.inertiaTensor, Is.EqualTo(float3.zero));
            }
            finally
            {
                bodyData.Dispose();
            }
        }

        [Test]
        public void RefreshPhysicsSlotClassifiesSolidCubeBoundary()
        {
            using var scope = new EntityDataTestScope();

            // Solid 3x3x3 cube hugging the origin corner; entirely inside brick (0,0,0).
            for (int z = 0; z < 3; z++)
            {
                for (int y = 0; y < 3; y++)
                {
                    for (int x = 0; x < 3; x++)
                    {
                        scope.Data.SetBlock(new int3(x, y, z), new Block(1));
                    }
                }
            }

            // Physics-slot generation is gated on the require-update (read) buffer that dirty
            // propagation would normally populate; mark it directly since no propagation runs here.
            // MarkRequired skips unallocated bricks, so it must follow the writes above.
            scope.Data.MarkRequired(int3.zero, BrickUpdateFlags.GeometryWithLocalNeighbor);
            scope.Data.RefreshNonEmptyMask();

            var bodyData = new VoxelBodyData(Allocator.Persistent);
            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);

                Assert.That(UnsafeUtility.SizeOf<PhysicsInfo>(), Is.EqualTo(1));

                // The minimum corner is a geometric corner of the box, so every surface cell rooted
                // there is active at once: three boundary faces, three convex edges and the point.
                // The cube exists too, which is what makes all eight bits set.
                Assert.That(PhysicsData(scope, 0, 0, 0).data, Is.EqualTo(0xFF),
                    "Minimum corner roots every cell, including all seven surface features");

                // A voxel on a box edge keeps the edge running along that box edge, the two boundary
                // faces meeting there, and its cube. Its point is absorbed by the collinear edges.
                Assert.That(PhysicsData(scope, 1, 0, 0).data, Is.EqualTo(
                    (1 << PhysicsInfo.BitEdgeX) | (1 << PhysicsInfo.BitFaceXY) |
                    (1 << PhysicsInfo.BitFaceXZ) | (1 << PhysicsInfo.BitCube)));

                // A voxel in the middle of a flat boundary face keeps only that face. Its edges are
                // flat subdivisions and its point is interior to the face.
                Assert.That(PhysicsData(scope, 1, 1, 0).data, Is.EqualTo(
                    (1 << PhysicsInfo.BitFaceXY) | (1 << PhysicsInfo.BitCube)));

                // The centre voxel is deep inside solid: no surface feature at all, only the volume
                // cube. This is the case IsInterior now names.
                Assert.That(PhysicsData(scope, 1, 1, 1).data,
                    Is.EqualTo(1 << PhysicsInfo.BitCube));
                Assert.That(PhysicsData(scope, 1, 1, 1).IsInterior, Is.True);
                Assert.That(PhysicsData(scope, 1, 1, 1).HasVolumeCell, Is.True);

                // The maximum side roots no cell that grows forward, but it is still real boundary.
                // A max-face voxel keeps the two in-plane edges and the face they bound; the max
                // corner keeps only its point. Under containment dedup all three read as zero.
                Assert.That(PhysicsData(scope, 2, 0, 0).data, Is.EqualTo(
                    (1 << PhysicsInfo.BitPoint) | (1 << PhysicsInfo.BitEdgeY) |
                    (1 << PhysicsInfo.BitEdgeZ) | (1 << PhysicsInfo.BitFaceYZ)));
                Assert.That(PhysicsData(scope, 2, 2, 0).data, Is.EqualTo(
                    (1 << PhysicsInfo.BitPoint) | (1 << PhysicsInfo.BitEdgeZ)));
                Assert.That(PhysicsData(scope, 2, 2, 2).data,
                    Is.EqualTo(1 << PhysicsInfo.BitPoint));
                Assert.That(PhysicsData(scope, 2, 2, 2).IsInterior, Is.False);

                // Keys are the roots carrying an active point or edge, i.e. the twelve box edge
                // chains and the eight corners. The six face centres and the centre are not keys.
                Assert.That(IsPhysicsKey(scope, 0, 0, 0), Is.True);
                Assert.That(IsPhysicsKey(scope, 2, 2, 2), Is.True);
                Assert.That(IsPhysicsKey(scope, 1, 0, 0), Is.True);
                Assert.That(IsPhysicsKey(scope, 1, 1, 0), Is.False);
                Assert.That(IsPhysicsKey(scope, 1, 1, 1), Is.False);

                // Air block inside the allocated brick is cleared, not stale.
                Assert.That(PhysicsData(scope, 5, 5, 5).data, Is.EqualTo(0));
            }
            finally
            {
                bodyData.Dispose();
            }
        }

        [Test]
        public void RefreshPhysicsSlotRootsIsolatedVoxelAsPoint()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(new int3(1, 1, 1), new Block(1));
            scope.Data.MarkRequired(int3.zero, BrickUpdateFlags.GeometryWithLocalNeighbor);
            scope.Data.RefreshNonEmptyMask();

            var bodyData = new VoxelBodyData(Allocator.Persistent);
            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);

                // With no face neighbor the bare point is the only cell, and it is the whole body.
                Assert.That(PhysicsData(scope, 1, 1, 1).data,
                    Is.EqualTo(1 << PhysicsInfo.BitPoint));
                Assert.That(PhysicsData(scope, 1, 1, 1).HasPointFeature, Is.True);
                Assert.That(IsPhysicsKey(scope, 1, 1, 1), Is.True);
            }
            finally
            {
                bodyData.Dispose();
            }
        }

        [Test]
        public void RefreshPhysicsSlotKeepsBothWireEndpoints()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(new int3(1, 1, 1), new Block(1));
            scope.Data.SetBlock(new int3(2, 1, 1), new Block(1));
            scope.Data.MarkRequired(int3.zero, BrickUpdateFlags.GeometryWithLocalNeighbor);
            scope.Data.RefreshNonEmptyMask();

            var bodyData = new VoxelBodyData(Allocator.Persistent);
            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);

                // The segment carries the geometry, but both endpoints stay active: an endpoint owns
                // the cap of directions past the end of the segment, which the segment does not.
                // Those two points are what lets a wire form a permitted pair against a face.
                Assert.That(PhysicsData(scope, 1, 1, 1).data, Is.EqualTo(
                    (1 << PhysicsInfo.BitEdgeX) | (1 << PhysicsInfo.BitPoint)));
                Assert.That(PhysicsData(scope, 2, 1, 1).data,
                    Is.EqualTo(1 << PhysicsInfo.BitPoint));

                // Both roots carry a point or an edge, so both are contact sources.
                Assert.That(IsPhysicsKey(scope, 1, 1, 1), Is.True);
                Assert.That(IsPhysicsKey(scope, 2, 1, 1), Is.True);
            }
            finally
            {
                bodyData.Dispose();
            }
        }

        [Test]
        public void RefreshPhysicsSlotKeepsBothArmsOfAnLCorner()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(new int3(1, 1, 1), new Block(1));
            scope.Data.SetBlock(new int3(2, 1, 1), new Block(1));
            scope.Data.SetBlock(new int3(1, 2, 1), new Block(1));
            scope.Data.MarkRequired(int3.zero, BrickUpdateFlags.GeometryWithLocalNeighbor);
            scope.Data.RefreshNonEmptyMask();

            var bodyData = new VoxelBodyData(Allocator.Persistent);
            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);

                // No square exists, so both segments stay on the corner voxel and no diagonal square
                // is fabricated. The corner point is active too - the two arms are not collinear, so
                // they cannot absorb it - and each arm end keeps its own endpoint.
                Assert.That(PhysicsData(scope, 1, 1, 1).data, Is.EqualTo(
                    (1 << PhysicsInfo.BitEdgeX) | (1 << PhysicsInfo.BitEdgeY) |
                    (1 << PhysicsInfo.BitPoint)));
                Assert.That(PhysicsData(scope, 2, 1, 1).data,
                    Is.EqualTo(1 << PhysicsInfo.BitPoint));
                Assert.That(PhysicsData(scope, 1, 2, 1).data,
                    Is.EqualTo(1 << PhysicsInfo.BitPoint));
            }
            finally
            {
                bodyData.Dispose();
            }
        }

        [Test]
        public void PhysicsKeyMaskMatchesActivePointAndEdgeBits()
        {
            using var scope = new EntityDataTestScope();

            // A mix that produces every class at once: solid volume with a buried interior, a
            // one-thick plate, a wire with two ends, an L with two arms, and an isolated voxel.
            for (int z = 0; z < 3; z++)
            {
                for (int y = 0; y < 3; y++)
                {
                    for (int x = 0; x < 3; x++)
                    {
                        scope.Data.SetBlock(new int3(x, y, z), new Block(1));
                    }
                }
            }
            for (int y = 0; y < 5; y++)
            {
                for (int x = 0; x < 5; x++)
                {
                    scope.Data.SetBlock(new int3(x, y, 5), new Block(1));
                }
            }
            for (int x = 0; x < 5; x++)
            {
                scope.Data.SetBlock(new int3(x, 7, 7), new Block(1));
            }
            scope.Data.SetBlock(new int3(5, 3, 0), new Block(1));
            scope.Data.SetBlock(new int3(6, 3, 0), new Block(1));
            scope.Data.SetBlock(new int3(5, 4, 0), new Block(1));
            scope.Data.SetBlock(new int3(7, 0, 7), new Block(1));

            scope.Data.MarkRequired(int3.zero, BrickUpdateFlags.GeometryWithLocalNeighbor);
            scope.Data.RefreshNonEmptyMask();

            var bodyData = new VoxelBodyData(Allocator.Persistent);
            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);

                // The key bit must be exactly "roots an active point or an active edge". The source
                // enumeration relies on that to find every contact source, and the edge-edge query
                // relies on the superset half of it: it scans the key mask instead of occupancy, so
                // a root carrying an active edge whose key bit was clear would be a lost contact
                // that no scene-level test would localise.
                const int pointOrEdge = PhysicsInfo.PointMask | PhysicsInfo.EdgeMask;
                int checkedVoxels = 0;
                int keyVoxels = 0;

                for (int z = 0; z < BrickKey.BlocksPerAxis; z++)
                {
                    for (int y = 0; y < BrickKey.BlocksPerAxis; y++)
                    {
                        for (int x = 0; x < BrickKey.BlocksPerAxis; x++)
                        {
                            bool rootsPointOrEdge = (PhysicsData(scope, x, y, z).data & pointOrEdge) != 0;
                            bool isKey = IsPhysicsKey(scope, x, y, z);
                            Assert.That(isKey, Is.EqualTo(rootsPointOrEdge),
                                $"key bit disagrees with the point/edge bits at ({x},{y},{z})");

                            checkedVoxels++;
                            if (isKey)
                            {
                                keyVoxels++;
                            }
                        }
                    }
                }

                Assert.That(checkedVoxels, Is.EqualTo(BrickKey.BlocksInBrick));

                // Guards against the assertion passing vacuously on an all-clear brick, and against
                // a key mask that simply marks everything.
                Assert.That(keyVoxels, Is.GreaterThan(0), "the shape produced no keys at all");
                Assert.That(keyVoxels, Is.LessThan(checkedVoxels), "every voxel came back a key");
            }
            finally
            {
                bodyData.Dispose();
            }
        }

        [Test]
        public unsafe void PhysicsKeyEnumeratorFollowsPhysicsInfoBitmapInVoxelIndexOrder()
        {
            using var scope = new EntityDataTestScope();

            // In a 3x3x3 cube the keys are the roots carrying an active point or edge: the eight
            // geometric corners and the twelve box edge chains. The six face centres keep only a
            // flat face and the centre keeps only the volume cube, so neither is a key. That is
            // exactly "at least two coordinates on an extreme", i.e. 8 + 12 = 20 of the 27.
            for (int z = 0; z < 3; z++)
            {
                for (int y = 0; y < 3; y++)
                {
                    for (int x = 0; x < 3; x++)
                    {
                        scope.Data.SetBlock(new int3(x, y, z), new Block(1));
                    }
                }
            }

            scope.Data.MarkRequired(int3.zero, BrickUpdateFlags.GeometryWithLocalNeighbor);
            scope.Data.RefreshNonEmptyMask();

            var bodyData = new VoxelBodyData(Allocator.Persistent);
            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);
                Assert.That(
                    scope.Data.TryBindBrickAux(SectorSlotId.PhysicsInfo, int3.zero, out void* keyAux),
                    Is.True,
                    "the physics-key aux bitmap must exist after the refresh");
                Assert.That(keyAux != null, Is.True);

                BrickBitmaskSlotEnumerator<PhysicsInfo> enumerator =
                    scope.Data.EnumerateBrickBitmask<PhysicsInfo>(SectorSlotId.PhysicsInfo, int3.zero);
                int selected = 0;

                // Walk in flat voxel-index order (x fastest, then y, then z) and require the
                // enumerator to visit exactly the expected roots in that order.
                for (int z = 0; z < 3; z++)
                {
                    for (int y = 0; y < 3; y++)
                    {
                        for (int x = 0; x < 3; x++)
                        {
                            int extremes = (x == 1 ? 0 : 1) + (y == 1 ? 0 : 1) + (z == 1 ? 0 : 1);
                            if (extremes < 2)
                            {
                                Assert.That(IsPhysicsKey(scope, x, y, z), Is.False,
                                    $"({x},{y},{z}) is a face centre or the centre");
                                continue;
                            }

                            Assert.That(enumerator.MoveNext(), Is.True);
                            Assert.That(enumerator.Current.position, Is.EqualTo(new int3(x, y, z)));
                            Assert.That(enumerator.Current.value.IsInterior, Is.False);
                            selected++;
                        }
                    }
                }

                Assert.That(selected, Is.EqualTo(20));
                Assert.That(enumerator.MoveNext(), Is.False);

                // A fresh enumeration starts over at the first selected root.
                enumerator = scope.Data.EnumerateBrickBitmask<PhysicsInfo>(
                    SectorSlotId.PhysicsInfo, int3.zero);
                Assert.That(enumerator.MoveNext(), Is.True);
                Assert.That(enumerator.Current.position, Is.EqualTo(int3.zero));

                using var burstCount = new NativeArray<int>(1, Allocator.TempJob);
                new CountPhysicsKeyBlocksJob
                {
                    Entity = scope.Data,
                    Result = burstCount
                }.Schedule().Complete();
                Assert.That(burstCount[0], Is.EqualTo(20));
            }
            finally
            {
                bodyData.Dispose();
            }
        }

        private static PhysicsInfo PhysicsData(EntityDataTestScope scope, int x, int y, int z)
        {
            return scope.Data.GetSlot<PhysicsInfo>(SectorSlotId.PhysicsInfo, new int3(x, y, z));
        }

        private static unsafe bool IsPhysicsKey(EntityDataTestScope scope, int x, int y, int z)
        {
            int3 pos = new int3(x, y, z);
            if (!scope.Data.TryBindBrickAux(
                    SectorSlotId.PhysicsInfo, BrickKey.FromBlock(pos), out void* aux))
            {
                return false;
            }

            int3 local = BrickKey.LocalBlock(pos);
            return BrickBitmask.GetBit((ulong*)aux, BrickKey.ToBlockIdx(local.x, local.y, local.z));
        }

        [Test]
        public void PhysicsWorldBuildReadsPersistedMotionFromVoxelBodyData()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(int3.zero, new Block(1));
            scope.Data.RefreshNonEmptyMask();

            Guid128 guid = new Guid128(1, 2, 3, 4);
            var bodyData = new VoxelBodyData(Allocator.Persistent);
            var world = new PhysicsWorld(0, 0, 0);
            var tickBuf = new PhysicsStepInputs
            {
                VoxelEntities = new NativeHashMap<Guid128, VoxelEntityData>(1, Allocator.Persistent),
                VoxelBodies = new NativeHashMap<Guid128, VoxelBodyData>(1, Allocator.Persistent),
                nDynamicBodies = 1
            };
            NativeArray<Guid128> bodyIndexToGuid = default;

            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);
                bodyData._cached_body_index = 0;
                bodyData.motionData = new Unity.Physics.MotionData
                {
                    WorldFromMotion = RigidTransform.identity,
                    BodyFromMotion = RigidTransform.identity,
                    LinearDamping = 0.25f,
                    AngularDamping = 0.5f
                };
                bodyData.motionVelocity = new Unity.Physics.MotionVelocity
                {
                    LinearVelocity = new float3(1f, 2f, 3f),
                    AngularVelocity = new float3(4f, 5f, 6f),
                    InverseInertia = new float3(99f),
                    InverseMass = 99f,
                    AngularExpansionFactor = 7f,
                    GravityFactor = 0.25f
                };

                tickBuf.VoxelEntities.Add(guid, scope.Data);
                tickBuf.VoxelBodies.Add(guid, bodyData);

                // Global air friction (0.05 / 0.08 here) overrides each body's persisted MotionData
                // damping (0.25 / 0.5 above) so friction is a single live-tunable engine knob.
                JobHandle buildHandle = CaelixPhysicsInterface.SchedulePhysicsWorldBuild(
                    ref tickBuf,
                    ref world,
                    out bodyIndexToGuid,
                    0.05f,
                    0.08f,
                    default);
                buildHandle.Complete();

                Assert.That(bodyIndexToGuid[0], Is.EqualTo(guid));
                Assert.That(world.MotionDatas[0].LinearDamping, Is.EqualTo(0.05f),
                    "Global linear air friction must override the body's persisted LinearDamping");
                Assert.That(world.MotionDatas[0].AngularDamping, Is.EqualTo(0.08f),
                    "Global angular air friction must override the body's persisted AngularDamping");
                Assert.That(world.MotionVelocities[0].LinearVelocity, Is.EqualTo(new float3(1f, 2f, 3f)));
                Assert.That(world.MotionVelocities[0].AngularVelocity, Is.EqualTo(new float3(4f, 5f, 6f)));
                Assert.That(world.MotionVelocities[0].GravityFactor, Is.EqualTo(0.25f));
            }
            finally
            {
                if (bodyIndexToGuid.IsCreated)
                {
                    bodyIndexToGuid.Dispose();
                }

                world.Dispose();
                tickBuf.VoxelEntities.Dispose();
                tickBuf.VoxelBodies.Dispose();
                bodyData.Dispose();
            }
        }

        [Test]
        public void PhysicsWorldBuildUsesAbsoluteCachedIndicesForDynamicAndStaticBodies()
        {
            using var dynamicScope = new EntityDataTestScope();
            using var staticScope = new EntityDataTestScope();
            dynamicScope.Data.SetBlock(int3.zero, new Block(1));
            staticScope.Data.SetBlock(int3.zero, new Block(1));
            dynamicScope.Data.RefreshNonEmptyMask();
            staticScope.Data.RefreshNonEmptyMask();
            staticScope.Data.isStatic = true;

            Guid128 dynamicGuid = new Guid128(20, 21, 22, 23);
            Guid128 staticGuid = new Guid128(24, 25, 26, 27);
            var dynamicBody = new VoxelBodyData(Allocator.Persistent);
            var staticBody = new VoxelBodyData(Allocator.Persistent);
            var world = new PhysicsWorld(0, 0, 0);
            var tickBuf = new PhysicsStepInputs
            {
                VoxelEntities = new NativeHashMap<Guid128, VoxelEntityData>(2, Allocator.Persistent),
                VoxelBodies = new NativeHashMap<Guid128, VoxelBodyData>(2, Allocator.Persistent),
                nDynamicBodies = 1
            };
            NativeArray<Guid128> bodyIndexToGuid = default;

            try
            {
                dynamicBody.ComputePhysicsProperties(dynamicScope.Data);
                dynamicBody._cached_body_index = 0;
                staticBody.ComputePhysicsProperties(staticScope.Data);
                staticBody._cached_body_index = 1;

                tickBuf.VoxelEntities.Add(dynamicGuid, dynamicScope.Data);
                tickBuf.VoxelEntities.Add(staticGuid, staticScope.Data);
                tickBuf.VoxelBodies.Add(dynamicGuid, dynamicBody);
                tickBuf.VoxelBodies.Add(staticGuid, staticBody);

                JobHandle buildHandle = CaelixPhysicsInterface.SchedulePhysicsWorldBuild(
                    ref tickBuf,
                    ref world,
                    out bodyIndexToGuid,
                    0f,
                    0f,
                    default);
                buildHandle.Complete();

                Assert.That(bodyIndexToGuid[0], Is.EqualTo(dynamicGuid));
                Assert.That(bodyIndexToGuid[1], Is.EqualTo(staticGuid));
            }
            finally
            {
                if (bodyIndexToGuid.IsCreated)
                {
                    bodyIndexToGuid.Dispose();
                }

                world.Dispose();
                tickBuf.VoxelEntities.Dispose();
                tickBuf.VoxelBodies.Dispose();
                dynamicBody.Dispose();
                staticBody.Dispose();
            }
        }

        [Test]
        public void PhysicsWorldExportPersistsMotionForMultipleDynamicBodies()
        {
            using var firstScope = new EntityDataTestScope();
            using var secondScope = new EntityDataTestScope();
            firstScope.Data.SetBlock(int3.zero, new Block(1));
            secondScope.Data.SetBlock(int3.zero, new Block(1));
            firstScope.Data.RefreshNonEmptyMask();
            secondScope.Data.RefreshNonEmptyMask();

            Guid128 firstGuid = new Guid128(5, 6, 7, 8);
            Guid128 secondGuid = new Guid128(9, 10, 11, 12);
            var firstBody = new VoxelBodyData(Allocator.Persistent);
            var secondBody = new VoxelBodyData(Allocator.Persistent);
            var world = new PhysicsWorld(0, 0, 0);
            var tickBuf = new PhysicsStepInputs
            {
                VoxelEntities = new NativeHashMap<Guid128, VoxelEntityData>(2, Allocator.Persistent),
                VoxelBodies = new NativeHashMap<Guid128, VoxelBodyData>(2, Allocator.Persistent),
                nDynamicBodies = 2
            };
            NativeArray<Guid128> bodyIndexToGuid = default;
            bool exportScheduled = false;

            try
            {
                firstBody.ComputePhysicsProperties(firstScope.Data);
                firstBody._cached_body_index = 0;
                secondBody.ComputePhysicsProperties(secondScope.Data);
                secondBody._cached_body_index = 1;
                tickBuf.VoxelEntities.Add(firstGuid, firstScope.Data);
                tickBuf.VoxelEntities.Add(secondGuid, secondScope.Data);
                tickBuf.VoxelBodies.Add(firstGuid, firstBody);
                tickBuf.VoxelBodies.Add(secondGuid, secondBody);

                JobHandle buildHandle = CaelixPhysicsInterface.SchedulePhysicsWorldBuild(
                    ref tickBuf,
                    ref world,
                    out bodyIndexToGuid,
                    0f,
                    0f,
                    default);
                buildHandle.Complete();

                NativeArray<Unity.Physics.MotionData> motionDatas = world.MotionDatas;
                Unity.Physics.MotionData exportedMotionData = motionDatas[0];
                exportedMotionData.WorldFromMotion = new RigidTransform(quaternion.identity, new float3(10f, 20f, 30f));
                motionDatas[0] = exportedMotionData;
                Unity.Physics.MotionData secondExportedMotionData = motionDatas[1];
                secondExportedMotionData.WorldFromMotion =
                    new RigidTransform(quaternion.identity, new float3(40f, 50f, 60f));
                motionDatas[1] = secondExportedMotionData;

                NativeArray<Unity.Physics.MotionVelocity> motionVelocities = world.MotionVelocities;
                Unity.Physics.MotionVelocity exportedMotionVelocity = motionVelocities[0];
                exportedMotionVelocity.LinearVelocity = new float3(2f, 4f, 6f);
                exportedMotionVelocity.AngularVelocity = new float3(1f, 3f, 5f);
                exportedMotionVelocity.GravityFactor = 0.75f;
                motionVelocities[0] = exportedMotionVelocity;
                Unity.Physics.MotionVelocity secondExportedMotionVelocity = motionVelocities[1];
                secondExportedMotionVelocity.LinearVelocity = new float3(8f, 10f, 12f);
                secondExportedMotionVelocity.AngularVelocity = new float3(7f, 9f, 11f);
                secondExportedMotionVelocity.GravityFactor = 0.5f;
                motionVelocities[1] = secondExportedMotionVelocity;

                JobHandle exportHandle = CaelixPhysicsInterface.SchedulePhysicsWorldExport(
                    ref tickBuf,
                    ref world,
                    bodyIndexToGuid,
                    default);
                exportScheduled = true;
                exportHandle.Complete();

                VoxelBodyData exportedFirstBody = tickBuf.VoxelBodies[firstGuid];
                Assert.That(exportedFirstBody.motionData.WorldFromMotion.pos,
                    Is.EqualTo(new float3(10f, 20f, 30f)));
                Assert.That(exportedFirstBody.motionVelocity.LinearVelocity, Is.EqualTo(new float3(2f, 4f, 6f)));
                Assert.That(exportedFirstBody.motionVelocity.AngularVelocity, Is.EqualTo(new float3(1f, 3f, 5f)));
                Assert.That(exportedFirstBody.motionVelocity.GravityFactor, Is.EqualTo(0.75f));

                VoxelBodyData exportedSecondBody = tickBuf.VoxelBodies[secondGuid];
                Assert.That(exportedSecondBody.motionData.WorldFromMotion.pos,
                    Is.EqualTo(new float3(40f, 50f, 60f)));
                Assert.That(exportedSecondBody.motionVelocity.LinearVelocity, Is.EqualTo(new float3(8f, 10f, 12f)));
                Assert.That(exportedSecondBody.motionVelocity.AngularVelocity, Is.EqualTo(new float3(7f, 9f, 11f)));
                Assert.That(exportedSecondBody.motionVelocity.GravityFactor, Is.EqualTo(0.5f));
            }
            finally
            {
                if (!exportScheduled && bodyIndexToGuid.IsCreated)
                {
                    bodyIndexToGuid.Dispose();
                }

                world.Dispose();
                tickBuf.VoxelEntities.Dispose();
                tickBuf.VoxelBodies.Dispose();
                firstBody.Dispose();
                secondBody.Dispose();
            }
        }

        [Test]
        public void BodyForceCommandStreamAppliesMainThreadForceBeforePhysicsBuild()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(int3.zero, new Block(1));
            scope.Data.RefreshNonEmptyMask();

            Guid128 guid = new Guid128(9, 10, 11, 12);
            var bodyData = new VoxelBodyData(Allocator.Persistent);
            var tickBuf = new PhysicsStepInputs
            {
                VoxelEntities = new NativeHashMap<Guid128, VoxelEntityData>(1, Allocator.Persistent),
                VoxelBodies = new NativeHashMap<Guid128, VoxelBodyData>(1, Allocator.Persistent)
            };
            var commands = new VoxelBodyForceCommandStream(Allocator.Persistent);

            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);
                tickBuf.VoxelEntities.Add(guid, scope.Data);
                tickBuf.VoxelBodies.Add(guid, bodyData);

                commands.AddForce(guid, new float3(4f, 0f, 0f), VoxelBodyForceMode.Force);
                commands.ApplyTo(ref tickBuf, 0.5f);

                VoxelBodyData updatedBody = tickBuf.VoxelBodies[guid];
                Assert.That(updatedBody.motionVelocity.LinearVelocity, Is.EqualTo(new float3(2f, 0f, 0f)));
            }
            finally
            {
                commands.Dispose();
                tickBuf.VoxelEntities.Dispose();
                tickBuf.VoxelBodies.Dispose();
                bodyData.Dispose();
            }
        }

        [Test]
        public void BodyForceCommandStreamAppliesOffCenterImpulseTorque()
        {
            using var scope = new EntityDataTestScope();
            scope.Data.SetBlock(int3.zero, new Block(1));
            scope.Data.SetBlock(new int3(2, 0, 0), new Block(1));
            scope.Data.RefreshNonEmptyMask();

            Guid128 guid = new Guid128(13, 14, 15, 16);
            var bodyData = new VoxelBodyData(Allocator.Persistent);
            var tickBuf = new PhysicsStepInputs
            {
                VoxelEntities = new NativeHashMap<Guid128, VoxelEntityData>(1, Allocator.Persistent),
                VoxelBodies = new NativeHashMap<Guid128, VoxelBodyData>(1, Allocator.Persistent)
            };
            var commands = new VoxelBodyForceCommandStream(Allocator.Persistent);

            try
            {
                bodyData.ComputePhysicsProperties(scope.Data);
                tickBuf.VoxelEntities.Add(guid, scope.Data);
                tickBuf.VoxelBodies.Add(guid, bodyData);

                // AsJobWriter now reserves capacity for N commands and returns a
                // ParallelWriter-backed writer (AddNoResize); no foreach-index bookkeeping.
                VoxelBodyForceCommandStream.JobWriter writer = commands.AsJobWriter(1);
                float3 centerOfMass = bodyData.massProperties.centerOfMass;
                writer.AddForceAtPosition(
                    guid,
                    new float3(2f, 0f, 0f),
                    centerOfMass + new float3(0f, 1f, 0f),
                    VoxelBodyForceMode.Impulse);
                commands.ApplyTo(ref tickBuf, 1f);

                VoxelBodyData updatedBody = tickBuf.VoxelBodies[guid];
                Assert.That(updatedBody.motionVelocity.LinearVelocity, Is.EqualTo(new float3(1f, 0f, 0f)));
                Assert.That(updatedBody.motionVelocity.AngularVelocity, Is.EqualTo(new float3(0f, 0f, -1f)));
            }
            finally
            {
                commands.Dispose();
                tickBuf.VoxelEntities.Dispose();
                tickBuf.VoxelBodies.Dispose();
                bodyData.Dispose();
            }
        }
    }
}
