using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Systems;
using Unity.Transforms;
using UnityEngine;

namespace Unity.Physics.Tests.Dynamics
{
    class EventsTestBase
    {
        BlobAssetReference<Collider> Collider;
        protected Unity.Physics.Simulation m_EmptySimulation;
        protected SimulationSingleton m_EmptySimulationSingleton;
        protected PhysicsWorld m_EmptyWorld;

        [SetUp]
        protected virtual void SetUp()
        {
            Collider = SphereCollider.Create(new SphereGeometry {Radius = 0.25f});

            m_EmptySimulation = Unity.Physics.Simulation.Create();
            m_EmptySimulationSingleton = new SimulationSingleton();
            m_EmptySimulationSingleton.InitializeFromSimulation(ref m_EmptySimulation);
            m_EmptyWorld = new PhysicsWorld(10, 10, 10);
        }

        [TearDown]
        protected virtual void TearDown()
        {
            Collider.Dispose();

            m_EmptyWorld.Dispose();
            m_EmptySimulation.Dispose();
        }

        public enum SolverTypes
        {
            OnlyIterative,
            OnlyDirect,
            MixedCoupled,
            MixedNotCoupled
        }

        public enum EventType
        {
            Collision,
            Trigger,
            Impulse
        }

        /// <summary>
        /// Creates a number of separate groups of three overlapping rigid bodies. Each group will produce exactly 3 collision or trigger events.
        /// <returns> Number of expected collision or trigger events. </returns>
        /// </summary>
        static int CreateRigidBodiesCausingEvents(Entities.World world, int numBodyGroups, BlobAssetReference<Collider> collider,
            SolverTypes solverTypes, bool exportCollisionEvents)
        {
            // enable events on collider:
            collider.Value.SetCollisionResponse(exportCollisionEvents ? CollisionResponsePolicy.CollideRaiseCollisionEvents
                : CollisionResponsePolicy.RaiseTriggerEvents);

            for (int group = 0; group < numBodyGroups; ++group)
            {
                // create 3 bodies with colliders, either "1 static and two dynamic" or "1 static, 1 dynamic and 1 kinematic",
                // all located at the same place, causing the following overlaps.
                // Either:
                // - two static-dynamic overlaps
                // - one dynamic-dynamic overlap
                // Or:
                //  - one static-dynamic overlap
                //  - one static-kinematic overlap
                //  - one dynamic-kinematic overlap
                // Each of these overlaps is expected to yield a collision or trigger event, that is, in total we expect
                // 3 * numGroups events to be generated.
                for (int body = 0; body < 3; ++body)
                {
                    var bodyIsStatic = body == 0;
                    var entity = bodyIsStatic

                        // static body:
                        ? world.EntityManager.CreateEntity(typeof(PhysicsCollider), typeof(LocalTransform),
                            typeof(PhysicsWorldIndex), typeof(PhysicsSolverType))

                        // dynamic or kinematic body:
                        : world.EntityManager.CreateEntity(typeof(PhysicsCollider), typeof(LocalTransform),
                            typeof(PhysicsWorldIndex), typeof(PhysicsSolverType), typeof(PhysicsVelocity), typeof(PhysicsMass));

                    world.EntityManager.SetComponentData(entity, new PhysicsCollider { Value = collider });

                    // place all entities at location (i, 0, 0)
                    world.EntityManager.SetComponentData(entity,
                        new LocalTransform { Position = new float3(group, 0, 0), Rotation = quaternion.identity, Scale = 1.0f });

                    if (!bodyIsStatic)
                    {
                        var bodyIsKinematic = group % 2 == 1 && body != 1; // one of the two dynamic bodies is kinematic in every second group
                        world.EntityManager.SetComponentData(entity,
                            bodyIsKinematic ? PhysicsMass.CreateKinematic(collider.Value.MassProperties)
                                : PhysicsMass.CreateDynamic(collider.Value.MassProperties, 1f));
                    }

                    SolverType solverType;
                    if (solverTypes == SolverTypes.MixedCoupled)
                    {
                        // Cause coupling between direct and iterative solver by making either the static or one of the
                        // dynamic bodies iterative and the rest direct, cycling through all combinations over the groups.
                        int iterativeBodyIndex = group % 3;
                        solverType = body == iterativeBodyIndex ? SolverType.Iterative : SolverType.Direct;
                    }
                    else if (solverTypes == SolverTypes.MixedNotCoupled)
                    {
                        // Alternative between groups of direct and iterative bodies.
                        solverType = group % 2 == 0 ? SolverType.Direct : SolverType.Iterative;
                    }
                    else
                    {
                        // always assign the same solver type to all
                        solverType = solverTypes == SolverTypes.OnlyDirect ? SolverType.Direct : SolverType.Iterative;
                    }

                    world.EntityManager.SetComponentData(entity,
                        new PhysicsSolverType { Value = solverType });
                }
            }

            return 3 * numBodyGroups;
        }

        /// <summary>
        /// Creates a number of separate groups of three jointed rigid bodies. Each group will produce exactly 2 impulse events.
        /// <returns> Number of expected events. </returns>
        /// </summary>
        static unsafe int CreateJointedBodiesCausingEvents(Entities.World world, int numBodyGroups, SolverTypes solverTypes)
        {
            Entity* bodies = stackalloc Entity[3];
            for (int group = 0; group < numBodyGroups; ++group)
            {
                // Create 3 bodies, either "1 static and two dynamic" or "1 static, 1 dynamic and 1 kinematic",
                // all located at the same place.
                // Then attach the static body to the first dynamic body, and the dynamic body to the second dynamic body (or
                // to the kinematic body) with springs.
                // Finally, enable impulse events so that each group produces exactly 2 impulse events.
                for (int body = 0; body < 3; ++body)
                {
                    var bodyIsStatic = body == 0;
                    var entity = bodyIsStatic

                        // static body:
                        ? world.EntityManager.CreateEntity(typeof(LocalTransform), typeof(PhysicsCollider),
                            typeof(PhysicsWorldIndex))

                        // dynamic or kinematic body:
                        : world.EntityManager.CreateEntity(typeof(LocalTransform),
                            typeof(PhysicsWorldIndex), typeof(PhysicsVelocity), typeof(PhysicsMass));

                    // place all entities at location (i, 0, 0) to ensure violation of constraints
                    world.EntityManager.SetComponentData(entity,
                        new LocalTransform { Position = new float3(group, 0, 0), Rotation = quaternion.identity, Scale = 1.0f });

                    if (!bodyIsStatic)
                    {
                        var bodyIsKinematic = group % 2 == 1 && body != 1; // one of the two dynamic bodies is kinematic in every second group
                        world.EntityManager.SetComponentData(entity,
                            bodyIsKinematic ? PhysicsMass.CreateKinematic(MassProperties.CreateSphere(1))
                                : PhysicsMass.CreateDynamic(MassProperties.CreateSphere(1), 1f));
                    }

                    bodies[body] = entity;
                }

                // Create spring joint component with some non-zero rest length
                var jointComponent = PhysicsJoint.CreateLimitedDistance(float3.zero, float3.zero, new Math.FloatRange(0.1f, 0.1f));

                // Enable impulse events
                jointComponent.SetImpulseEventThresholdAllConstraints(float3.zero, float3.zero);

                // create joints
                for (int joint = 0; joint < 2; ++joint)
                {
                    var jointEntity = world.EntityManager.CreateEntity(typeof(PhysicsJoint),
                        typeof(PhysicsConstrainedBodyPair), typeof(PhysicsWorldIndex), typeof(PhysicsSolverType));

                    world.EntityManager.SetComponentData(jointEntity, new PhysicsConstrainedBodyPair
                    {
                        Entities = new EntityPair {EntityA = bodies[joint], EntityB = bodies[joint+1]}
                    });

                    world.EntityManager.SetComponentData(jointEntity, jointComponent);

                    SolverType solverType;
                    if (solverTypes == SolverTypes.MixedCoupled)
                    {
                        // Cause coupling between direct and iterative solver by making one of the two joints direct
                        // alternated over the groups.
                        int iterativeJointIndex = group % 2;
                        solverType = joint == iterativeJointIndex ? SolverType.Iterative : SolverType.Direct;
                    }
                    else if (solverTypes == SolverTypes.MixedNotCoupled)
                    {
                        // Alternate between groups of direct and iterative joints.
                        solverType = group % 2 == 0 ? SolverType.Direct : SolverType.Iterative;
                    }
                    else
                    {
                        // always assign the same solver type to all joints
                        solverType = solverTypes == SolverTypes.OnlyDirect ? SolverType.Direct : SolverType.Iterative;
                    }

                    world.EntityManager.SetComponentData(jointEntity,
                        new PhysicsSolverType { Value = solverType });
                }
            }

            return 2 * numBodyGroups;
        }

        /// <summary>
        /// Creates a physics world containing a given number of groups of overlapping rigid bodies which will
        /// produce 'expectedCollisionEventCount' many collision events.
        /// </summary>
        public PhysicsWorld CreatePhysicsWorldCausingEvents(Entities.World world, int numBodyGroups,
            SolverTypes solverTypes, EventType eventType, out int expectedEventCount)
        {
            if (eventType != EventType.Impulse)
            {
                // create rigid body entities with overlaps
                expectedEventCount = CreateRigidBodiesCausingEvents(world, numBodyGroups, Collider, solverTypes,
                    exportCollisionEvents: eventType == EventType.Collision);
            }
            else
            {
                // create jointed body entities
                expectedEventCount = CreateJointedBodiesCausingEvents(world, numBodyGroups, solverTypes);
            }

            // create and update BuildPhysicsWorld system
            var buildPhysicsWorld = world.GetOrCreateSystem<BuildPhysicsWorld>();
            buildPhysicsWorld.Update(world.Unmanaged);
            var jobHandle = world.Unmanaged.ResolveSystemStateRef(buildPhysicsWorld).Dependency;
            jobHandle.Complete();
            Assert.IsTrue(jobHandle.IsCompleted);

            // obtain and return physics world:
            var worldData = world.EntityManager.GetComponentData<BuildPhysicsWorldData>(buildPhysicsWorld);
            return worldData.PhysicsData.PhysicsWorld;
        }

        /// <summary>
        /// Creates a non-empty physics world containing some dynamic rigid bodies without overlaps which will produce no events.
        /// </summary>
        public PhysicsWorld CreatePhysicsWorldCausingNoEvents(Entities.World world)
        {
            int numBodyGroups = 10;
            for (int i = 0; i < numBodyGroups; ++i)
            {
                var entity = world.EntityManager.CreateEntity(typeof(PhysicsCollider), typeof(LocalTransform),
                    typeof(PhysicsWorldIndex), typeof(PhysicsVelocity), typeof(PhysicsMass));

                world.EntityManager.SetComponentData(entity,
                    new LocalTransform { Position = new float3(i, 0, 0), Rotation = quaternion.identity, Scale = 1.0f });

                world.EntityManager.SetComponentData(entity, PhysicsMass.CreateDynamic(MassProperties.UnitSphere, 1f));
            }

            // create and update BuildPhysicsWorld system
            var buildPhysicsWorld = world.GetOrCreateSystem<BuildPhysicsWorld>();
            buildPhysicsWorld.Update(world.Unmanaged);
            var jobHandle = world.Unmanaged.ResolveSystemStateRef(buildPhysicsWorld).Dependency;
            jobHandle.Complete();
            Assert.IsTrue(jobHandle.IsCompleted);

            // obtain and return physics world:
            var worldData = world.EntityManager.GetComponentData<BuildPhysicsWorldData>(buildPhysicsWorld);
            return worldData.PhysicsData.PhysicsWorld;
        }

        static int GetEventCount(in Unity.Physics.Simulation simulation, EventType eventType)
        {
            var numEvents = 0;
            switch (eventType)
            {
                case EventType.Collision:
                    var collisionEvents = simulation.CollisionEvents;
                    foreach (var _ in collisionEvents)
                    {
                        ++numEvents;
                    }
                    break;
                case EventType.Trigger:
                    var triggerEvents = simulation.TriggerEvents;
                    foreach (var _ in triggerEvents)
                    {
                        ++numEvents;
                    }
                    break;
                case EventType.Impulse:
                    var impulseEvents = simulation.ImpulseEvents;
                    foreach (var _ in impulseEvents)
                    {
                        ++numEvents;
                    }
                    break;
            }

            return numEvents;
        }

        static void RunEventCountJob(in NativeArray<int> eventCount, NativeParallelHashSet<int2> bodyPairsSet,
            EventType eventType, in SimulationSingleton simulationSingleton, int numBodies, bool parallelEventProcessing)
        {
            JobHandle jobHandle = default;
            const int kInnerLoopBatchCount = 4;
            switch (eventType)
            {
                case EventType.Collision:
                {
                    var job = new CollisionEventsCountJob
                    {
                        EventCount = eventCount,
                        BodyPairsSet = bodyPairsSet.AsParallelWriter(),
                        NumBodies = numBodies
                    };
                    jobHandle = parallelEventProcessing ? job.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, default)
                        : job.Schedule(simulationSingleton, default);
                    break;
                }
                case EventType.Trigger:
                {
                    var job = new TriggerEventsCountJob
                    {
                        EventCount = eventCount,
                        BodyPairsSet = bodyPairsSet.AsParallelWriter(),
                        NumBodies = numBodies
                    };
                    jobHandle = parallelEventProcessing ? job.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, default)
                        : job.Schedule(simulationSingleton, default);
                    break;
                }
                case EventType.Impulse:
                {
                    var job = new ImpulseEventsCountJob
                    {
                        EventCount = eventCount,
                        BodyPairsSet = bodyPairsSet.AsParallelWriter(),
                        NumBodies = numBodies
                    };
                    jobHandle = parallelEventProcessing ? job.ScheduleParallel(kInnerLoopBatchCount, simulationSingleton, default)
                        : job.Schedule(simulationSingleton, default);
                    break;
                }
            }

            jobHandle.Complete();
        }

        static void ValidateEvent(int bodyIndexA, int bodyIndexB, int numBodies,
            NativeParallelHashSet<int2>.ParallelWriter bodyPairsSet)
        {
            var range = new int2(0, numBodies - 1);
            SafetyChecks.CheckInRangeAndThrow(bodyIndexA, range, "BodyIndexA");
            SafetyChecks.CheckInRangeAndThrow(bodyIndexB, range, "BodyIndexB");

            // make sure each body pair is unique
            var added = bodyPairsSet.TryAdd(new int2(bodyIndexA, bodyIndexB));
            SafetyChecks.CheckAreEqualAndThrow(true, added);
        }

        [BurstCompile]
        struct CollisionEventsCountJob
            : ICollisionEventsJob
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int> EventCount;

            public NativeParallelHashSet<int2>.ParallelWriter BodyPairsSet;
            public int NumBodies;

            [NativeSetThreadIndex]
            int m_ThreadID;

            public void Execute(CollisionEvent collisionEvent)
            {
                ValidateEvent(collisionEvent.BodyIndexA, collisionEvent.BodyIndexB, NumBodies, BodyPairsSet);

                EventCount[m_ThreadID]++;
            }
        }

        [BurstCompile]
        struct ImpulseEventsCountJob
            : IImpulseEventsJob
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int> EventCount;

            public NativeParallelHashSet<int2>.ParallelWriter BodyPairsSet;
            public int NumBodies;

            [NativeSetThreadIndex]
            int m_ThreadID;

            public void Execute(ImpulseEvent impulseEvent)
            {
                ValidateEvent(impulseEvent.BodyIndexA, impulseEvent.BodyIndexB, NumBodies, BodyPairsSet);

                EventCount[m_ThreadID]++;
            }
        }

        [BurstCompile]
        struct TriggerEventsCountJob
            : ITriggerEventsJob
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int> EventCount;

            public NativeParallelHashSet<int2>.ParallelWriter BodyPairsSet;
            public int NumBodies;

            [NativeSetThreadIndex]
            int m_ThreadID;

            public void Execute(TriggerEvent triggerEvent)
            {
                ValidateEvent(triggerEvent.BodyIndexA, triggerEvent.BodyIndexB, NumBodies, BodyPairsSet);

                EventCount[m_ThreadID]++;
            }
        }

        protected void VerifyExportEvents_StepJobs(bool multiThreaded, SolverTypes solverTypes, EventType eventType, bool parallelEventProcessing)
        {
            using (var world = new Entities.World("Test world"))
            {
                // build the physics world:

                int numBodyGroups = 50;
                var physicsWorld = CreatePhysicsWorldCausingEvents(world, numBodyGroups, solverTypes, eventType, out var expectedEventCount);

                // step the simulation one frame:

                using var haveStaticBodiesChanged = new NativeReference<int>(1, Allocator.TempJob);
                var stepInput = new SimulationStepInput
                {
                    World = physicsWorld,
                    TimeStep = 1/60f,
                    Gravity = new float3(0, -9.81f, 0),
                    NumSolverIterations = 4,
                    NumSubsteps = 1,
                    DirectSolverSettings = Solver.DirectSolverSettings.Default,
                    HaveStaticBodiesChanged = haveStaticBodiesChanged
                };

                var simulation = Unity.Physics.Simulation.Create();
                var handles = simulation.ScheduleStepJobs(stepInput, default, multiThreaded);
                handles.FinalExecutionHandle.Complete();
                handles.FinalDisposeHandle.Complete();

                // check event results:
                var numEvents = GetEventCount(simulation, eventType);

                Assert.Greater(expectedEventCount, 0);
                Assert.AreEqual(expectedEventCount, numEvents);

                // verify event job:
                var simulationSingleton = new SimulationSingleton();
                simulationSingleton.InitializeFromSimulation(ref simulation);
                using var contactCount = new NativeArray<int>(JobsUtility.JobWorkerCount + 1, Allocator.TempJob);
                using var bodyPairsSet = new NativeParallelHashSet<int2>(expectedEventCount, Allocator.TempJob);

                RunEventCountJob(contactCount, bodyPairsSet, eventType, simulationSingleton,
                    physicsWorld.NumBodies, parallelEventProcessing);

                int totalEventCount = 0;
                for (int i = 0; i < contactCount.Length; ++i)
                {
                    totalEventCount += contactCount[i];
                }

                Assert.AreEqual(expectedEventCount, totalEventCount);
                simulation.Dispose();
            }
        }

        protected void VerifyExportEvents_StepImmediate(SolverTypes solverTypes, EventType eventType)
        {
            using (var world = new Entities.World("Test world"))
            {
                // build the physics world:

                int numBodyGroups = 10;
                var physicsWorld = CreatePhysicsWorldCausingEvents(world, numBodyGroups, solverTypes,
                    eventType, out var expectedEventCount);

                // step the simulation one frame:

                using var haveStaticBodiesChanged = new NativeReference<int>(1, Allocator.TempJob);
                var stepInput = new SimulationStepInput
                {
                    World = physicsWorld,
                    TimeStep = 1/60f,
                    Gravity = new float3(0, -9.81f, 0),
                    NumSolverIterations = 4,
                    NumSubsteps = 1,
                    DirectSolverSettings = Solver.DirectSolverSettings.Default,
                    HaveStaticBodiesChanged = haveStaticBodiesChanged
                };

                var simulation = Unity.Physics.Simulation.Create();
                simulation.ResetSimulationContext(stepInput);
                simulation.Step(stepInput);

                // check event results:
                var numEvents = GetEventCount(simulation, eventType);

                Assert.Greater(expectedEventCount, 0);
                Assert.AreEqual(expectedEventCount, numEvents);
            }
        }
    }
}
