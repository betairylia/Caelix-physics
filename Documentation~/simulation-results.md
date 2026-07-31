# Simulation results

After completion of the [PhysicsSimulationGroup](physics-pipeline.md), the simulation step is complete.
The results of the simulation are:
- New positions, rotations and velocities of dynamic bodies - stored inside a `PhysicsWorld`.
- Collision and trigger events - stored internally in `Simulation` streams.

After `PhysicsSimulationGroup` finishes, the final system of the `PhysicsSystemGroup` can update - [ExportPhysicsWorld](physics-pipeline.md).
Its role is copy the updated body positions, rotations and velocities from `PhysicsWorld` ([physics simulation data](physics-data-types.md)) to ECS data.

## Events

In addition to updated positions, rotations and velocities of bodies, the results of simulation also include events.
Events are stored in streams stored inside the `Simulation` struct.
You can access them directly from `Simulation` by calling (`SystemBase|SystemAPI|EntityQuery`).GetSingleton<`SimulationSingleton`>().AsSimulation().(`CollisionEvents|TriggerEvents`) and iterate through them.
The other approach is to use specialised jobs.
To schedule these jobs, you will need [SimulationSingleton](physics-singletons.md).
Get it using (`SystemBase|SystemAPI|EntityQuery`).GetSingleton<`SimulationSingleton`>(). (See below)

>**Note:** Events are valid after the `PhysicsSimulationGroup` has finished, and up until it starts in the next frame. Using them while `PhysicsSimulationGroup` is updating will lead to undefined behaviour.

### Collision events

Collision events are raised from colliders that opted in to this behaviour.
For each collision that involves those colliders, a collision event will be raised.
To access them directly, [see above](#events).
To access them through a specialised job, implement and schedule an [`ICollisionEventsJob`](xref:Unity.Physics.ICollisionEventsJob) as shown in the example below.

**Example:**
```csharp
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateBefore(typeof(PhysicsSimulationGroup))] // We are updating before `PhysicsSimulationGroup` - this means that we will get the events of the previous frame. For the events of the current frame, use [UpdateAfter(...)] instead.
public partial struct GetNumCollisionEventsSystem : ISystem
{
    [BurstCompile]
    public struct CountNumCollisionEvents : ICollisionEventsJob
    {
        public NativeReference<int> NumCollisionEvents;
        public void Execute(CollisionEvent collisionEvent)
        {
            NumCollisionEvents.Value++;
        }
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        NativeReference<int> numCollisionEvents = new NativeReference<int>(0, Allocator.TempJob);

        state.Dependency = new CountNumCollisionEvents
        {
            NumCollisionEvents = numCollisionEvents
        }.Schedule(SystemAPI.GetSingleton<SimulationSingleton>(), state.Dependency);

        // ...
    }
}

```

In the above example the `ICollisionEventsJob` is run on a single thread.
For processing of large numbers of collision events, or if the `ICollisionEventsJob` has a high computational cost, it can be beneficial to schedule the job to run on multiple threads for parallel event processing by using the method [`ICollisionEventsJob.ScheduleParallel()`](xref:Unity.Physics.ICollisionEventJobExtensions.ScheduleParallel``1(``0,System.Int32,Unity.Physics.SimulationSingleton,Unity.Jobs.JobHandle)) as shown below.

**Example:**
```csharp
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(PhysicsSimulationGroup))] // We are updating after `PhysicsSimulationGroup` - this means that we will get the events of the current frame.
public partial struct ProcessCollisionEventsSystem : ISystem
{
    [BurstCompile]
    public struct ProcessCollisionEventsJob : ICollisionEventsJob
    {
        // ...

        // Since this job is scheduled using ScheduleParallel(),
        // the execute function will be called in parallel from multiple threads.
        public void Execute(CollisionEvent collisionEvent)
        {
            // Perform processing of collision events here.
            // ...
        }
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        // Schedule the job for parallel collision event processing.
        state.Dependency = new ProcessCollisionEventsJob
        {
            // ...
        }.ScheduleParallel(innerLoopBatchCount: 1, SystemAPI.GetSingleton<SimulationSingleton>(), state.Dependency);
        
        // ...
    }
}
```

### Trigger events

Trigger events are raised from colliders that opted in this behaviour.
These colliders will not collide with anything, but will instead raise a trigger event once a collision should have happened.
To access them directly, [see above](#events).
To access them through a specialised job, implement and schedule an `ITriggerEventsJob` as shown in the example below.
See example below.

**Example:**
```csharp
[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(PhysicsSimulationGroup))] // We are updating after `PhysicsSimulationGroup` - this means that we will get the events of the current frame.
public partial struct GetNumTriggerEventsSystem : ISystem
{
    [BurstCompile]
    public struct CountNumTriggerEvents : ITriggerEventsJob
    {
        public NativeReference<int> NumTriggerEvents;
        public void Execute(TriggerEvent collisionEvent)
        {
            NumTriggerEvents.Value++;
        }
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        NativeReference<int> numTriggerEvents = new NativeReference<int>(0, Allocator.TempJob);

        state.Dependency = new CountNumTriggerEvents
        {
            NumTriggerEvents = numTriggerEvents
        }.Schedule(SystemAPI.GetSingleton<SimulationSingleton>(), state.Dependency);

        // ...
    }
}
```

The above example runs the scheduled `ITriggerEventsJob` on a single thread. If beneficial for performance, an `ITriggerEventsJob` can also be scheduled to run on multiple threads for parallel event processing by using the method [`ITriggerEventsJob.ScheduleParallel()`](xref:Unity.Physics.ITriggerEventJobExtensions.ScheduleParallel``1(``0,System.Int32,Unity.Physics.SimulationSingleton,Unity.Jobs.JobHandle)), analogous to the `ICollisionEventsJob` [above](#collision-events).
