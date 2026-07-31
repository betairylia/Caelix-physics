using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Extensions;
using Unity.Physics.Systems;
using Unity.Transforms;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Serialization;

namespace Unity.Physics.Tests
{
    public class SimulationValidationAuthoring : MonoBehaviour
    {
        [Header("General Settings")]

        [Tooltip("Enables simulation validation.")]
        public bool EnableValidation = false;
        [Tooltip("Time period during which any validation is performed as simulation time interval [start, end] in seconds. Specify -1 as end value for a validation that never ends (default).")]
        public float2 ValidationTimeRange = new(0, -1);

        [Header("Validation Types")]

        [Tooltip("Validates if joints behave as expected, by comparing relative body positions and orientations and their relative angular and linear velocities.")]
        public bool ValidateJointBehavior = false;
        [Tooltip("Validates that all rigid bodies are moving in accordance with the specified expected linear and angular velocity values," +
            "not exceeding the provided linear and angular velocity error tolerances.")]
        [FormerlySerializedAs("ValidateRigidBodiesAtRest")]
        public bool ValidateRigidBodyMotion = false;
        [Tooltip("Validates that each rigid body's individual kinetic energy is conserved over time, considering the specified linear and angular kinetic energy error tolerances.\n" +
            "Note: Rigid bodies that are subject to gravity are excluded in this validation.")]
        public bool ValidateKineticEnergyConservation = false;

        [Header("Tolerances")]

        [Tooltip("Linear velocity error tolerance in meters/s")]
        public float LinearVelocityErrorTolerance = 0.005f;
        [Tooltip("Angular velocity error tolerance in radians/s")]
        public float AngularVelocityErrorTolerance = 0.01f;
        [Tooltip("Position error tolerance in meters")]
        public float PositionErrorTolerance = 0.01f;
        [Tooltip("Orientation error tolerance in radians")]
        public float OrientationErrorTolerance = 0.01f;
        [Tooltip("Linear kinetic energy error tolerance in Joules")]
        public float LinearKineticEnergyTolerance = 0.001f;
        [Tooltip("Angular kinetic energy error tolerance in Joules")]
        public float AngularKineticEnergyTolerance = 0.001f;

        [Header("Expected Values")]

        [Tooltip("Expected world space linear velocity in meters/s for rigid body motion validation")]
        public float3 ExpectedLinearVelocity = float3.zero;
        [Tooltip("Expected world space angular velocity in radians/s for rigid body motion validation")]
        public float3 ExpectedAngularVelocity = float3.zero;
    }

    public class SimulationValidationBaker : Baker<SimulationValidationAuthoring>
    {
        public override void Bake(SimulationValidationAuthoring authoring)
        {
            var entity = GetEntity(TransformUsageFlags.Dynamic);
            AddComponent(entity, new SimulationValidationSettings()
            {
                EnableValidation = authoring.EnableValidation,
                ValidateJointBehavior = authoring.ValidateJointBehavior,
                ValidateRigidBodyMotion = authoring.ValidateRigidBodyMotion,
                ValidateKineticEnergyConservation = authoring.ValidateKineticEnergyConservation,
                LinearVelocityErrorTolerance = authoring.LinearVelocityErrorTolerance,
                AngularVelocityErrorTolerance = authoring.AngularVelocityErrorTolerance,
                PositionErrorTolerance = authoring.PositionErrorTolerance,
                OrientationErrorTolerance = authoring.OrientationErrorTolerance,
                LinearKineticEnergyTolerance = authoring.LinearKineticEnergyTolerance,
                AngularKineticEnergyTolerance = authoring.AngularKineticEnergyTolerance,
                ExpectedLinearVelocity = authoring.ExpectedLinearVelocity,
                ExpectedAngularVelocity = authoring.ExpectedAngularVelocity,
                ValidationTimeRange = authoring.ValidationTimeRange
            });
        }
    }
    public struct SimulationValidationSettings : IComponentData
    {
        public bool EnableValidation;
        public bool ValidateJointBehavior;
        public bool ValidateRigidBodyMotion;
        public bool ValidateKineticEnergyConservation;
        public float LinearVelocityErrorTolerance;
        public float AngularVelocityErrorTolerance;
        public float PositionErrorTolerance;
        public float OrientationErrorTolerance;
        public float LinearKineticEnergyTolerance;
        public float AngularKineticEnergyTolerance;
        public float3 ExpectedLinearVelocity;
        public float3 ExpectedAngularVelocity;
        public float2 ValidationTimeRange;
    }

    /// <summary>
    /// Validation of all PhysicsJoint objects in the simulation.
    ///
    /// The expected behavior corresponds to joints created with the
    /// joint creation functions in PhysicsJoint, e.g., CreatePrismatic, CreateHinge, etc.
    /// </summary>
    [BurstCompile]
    public partial struct ValidateJointBehaviorJob : IJobEntity
    {
        [NativeDisableUnsafePtrRestriction]
        public SimulationValidationSystem.ErrorCounter Errors;
        [ReadOnly] public ComponentLookup<LocalTransform> TransformLookup;
        [ReadOnly] public ComponentLookup<LocalToWorld> LocalToWorldLookup;

        [ReadOnly] public ComponentLookup<PhysicsVelocity> PhysicsVelocityLookup;
        [ReadOnly] public ComponentLookup<PhysicsMass> PhysicsMassLookup;

        [ReadOnly] public DynamicsWorld DynamicsWorld;
        [ReadOnly] public NativeArray<Joint> Joints;

        [ReadOnly] public float PositionErrorTol;
        [ReadOnly] public float PositionErrorTolSq;
        [ReadOnly] public float OrientationErrorTol;
        [ReadOnly] public float OrientationErrorTolCos;
        [ReadOnly] public float AngVelErrorTol;
        [ReadOnly] public float AngVelErrorTolSq;
        [ReadOnly] public float LinVelErrorTol;
        [ReadOnly] public float LinVelErrorTolSq;

        [GenerateTestsForBurstCompatibility]
        static void ValidateConstraintType(in Constraint constraint, in ConstraintType expectedType)
        {
            Assert.AreEqual(expectedType, constraint.Type, $"Validation ({expectedType}): unexpected constraint type '{constraint.Type}'.");
        }

        [GenerateTestsForBurstCompatibility]
        void Execute(in Entity entity, in PhysicsJoint joint, in PhysicsConstrainedBodyPair bodyPair)
        {
            var jointIndex = DynamicsWorld.GetJointIndex(entity);
            var dynamicsJoint = Joints[jointIndex];
            var bodyAIx = dynamicsJoint.BodyPair.BodyIndexA;
            var bodyBIx = dynamicsJoint.BodyPair.BodyIndexB;

            var bodyAIsStatic = bodyAIx < 0 || bodyAIx >= DynamicsWorld.NumMotions;
            var bodyBIsStatic = bodyBIx < 0 || bodyBIx >= DynamicsWorld.NumMotions;
            if (bodyAIsStatic && bodyBIsStatic)
            {
                return;
            }

            var bodyATransform = bodyPair.EntityA != Entity.Null
                ? (bodyAIsStatic
                    ? LocalTransform.FromMatrix(LocalToWorldLookup[bodyPair.EntityA].Value) : TransformLookup[bodyPair.EntityA])
                : LocalTransform.Identity;

            var bodyBTransform = bodyPair.EntityB != Entity.Null
                ? (bodyBIsStatic
                    ? LocalTransform.FromMatrix(LocalToWorldLookup[bodyPair.EntityB].Value) : TransformLookup[bodyPair.EntityB])
                : LocalTransform.Identity;

            var bodyAWorld = new RigidTransform(bodyATransform.Rotation, bodyATransform.Position);
            var bodyBWorld = new RigidTransform(bodyBTransform.Rotation, bodyBTransform.Position);

            var anchorALocal = joint.BodyAFromJoint;
            var anchorBLocal = joint.BodyBFromJoint;

            var rigidAnchorALocal = anchorALocal.AsRigidTransform();
            var rigidAnchorBLocal = anchorBLocal.AsRigidTransform();

            var anchorAWorld = math.mul(bodyAWorld, rigidAnchorALocal);
            var anchorBWorld = math.mul(bodyBWorld, rigidAnchorBLocal);

            // pose validation for PhysicsJoints
            switch (joint.JointType)
            {
                case JointType.BallAndSocket:
                {
                    var deltaPos = anchorAWorld.pos - anchorBWorld.pos;
                    var posErrorSq = math.lengthsq(deltaPos);
                    if (posErrorSq > PositionErrorTolSq)
                    {
                        Errors.Add($"Joint anchor position is violated by {math.sqrt(posErrorSq)} meters, which exceeds position error tolerance of {PositionErrorTol} meters.",
                            "BallAndSocket", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    break;
                }
                case JointType.Hinge:
                case JointType.LimitedHinge:
                case JointType.AngularVelocityMotor:
                case JointType.RotationalMotor:
                {
                    // obtain hinge axis (attached to body A)
                    byte hingeConstraintBlockIndex = (byte)(joint.JointType == JointType.Hinge ? 0 : 1);
                    var hingeConstraint = joint[hingeConstraintBlockIndex];
                    ValidateConstraintType(hingeConstraint, ConstraintType.Angular);
                    var hingeAxisIndex = hingeConstraint.FreeAxis2D;
                    var hingeAxis = new float3x3(anchorAWorld.rot)[hingeAxisIndex];

                    // make sure rotation happens about the hinge axis
                    var rotBToA = math.mul(math.inverse(anchorAWorld.rot), anchorBWorld.rot);
                    rotBToA = math.normalize(rotBToA);
                    ((Quaternion)rotBToA).ToAngleAxis(out var angle, out var actualRotationAxis);
                    // We can only get a meaningful rotation axis between the two anchors if there is some reasonable amount of delta rotation.
                    // Note: angle is in degrees here
                    var absAngle = math.abs(angle);
                    var epsValidationAngle = 10.0f;
                    if (absAngle > epsValidationAngle && absAngle < 360f - epsValidationAngle)
                    {
                        actualRotationAxis = math.mul(anchorAWorld.rot, actualRotationAxis);
                        actualRotationAxis = math.normalize(actualRotationAxis);

                        // make sure hinge axis is aligned in both anchor frames
                        var cosAngle = math.dot(actualRotationAxis, hingeAxis);
                        var absCosAngle = math.abs(cosAngle);
                        var epsCos = OrientationErrorTolCos;
                        if (absCosAngle < epsCos)
                        {
                            Errors.Add($"Hinge axis orientation violated by {math.acos(absCosAngle)} radians, which exceeds orientation error tolerance of {OrientationErrorTol} radians",
                                "Hinge or equivalent", bodyPair.EntityA, bodyPair.EntityB);
                        }
                    }

                    // Make sure anchor positions are sufficiently close, as the bodies rotate around them.
                    var deltaPos = anchorAWorld.pos - anchorBWorld.pos;
                    var posErrorSq = math.lengthsq(deltaPos);
                    if (posErrorSq > PositionErrorTolSq)
                    {
                        Errors.Add($"Joint anchor position is violated by {math.sqrt(posErrorSq)} meters, which exceeds position error tolerance of {PositionErrorTol} meters.",
                            "Hinge or equivalent", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    break;
                }
                case JointType.Fixed:
                {
                    // make sure anchor frames are aligned

                    // orientation
                    var relQ = math.mul(math.inverse(anchorAWorld.rot), anchorBWorld.rot);
                    relQ = math.normalize(relQ);
                    var angle = 2.0 * math.acos(relQ.value.w);
                    var cosAngle = math.cos(angle);
                    if (cosAngle < OrientationErrorTolCos)
                    {
                        Errors.Add($"Relative orientation violated by {angle} radians, which exceeds orientation error tolerance of {OrientationErrorTol} radians",
                            "Fixed", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    // position
                    var deltaPos = anchorAWorld.pos - anchorBWorld.pos;
                    var posErrorSq = math.lengthsq(deltaPos);
                    if (posErrorSq > PositionErrorTolSq)
                    {
                        Errors.Add($"Joint anchor position is violated by {math.sqrt(posErrorSq)} meters, which exceeds position error tolerance of {PositionErrorTol} meters.",
                            "Fixed", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    break;
                }
                case JointType.Prismatic:
                case JointType.PositionalMotor:
                {
                    var constrainedAxisIndex = -1;
                    if (joint.JointType == JointType.Prismatic)
                    {
                        var linearConstraint = joint[1];
                        ValidateConstraintType(linearConstraint, ConstraintType.Linear);
                        constrainedAxisIndex = linearConstraint.ConstrainedAxis1D;
                    }
                    else if (joint.JointType == JointType.PositionalMotor)
                    {
                        var motorConstraint = joint[0];
                        ValidateConstraintType(motorConstraint, ConstraintType.PositionMotor);
                        constrainedAxisIndex = motorConstraint.ConstrainedAxis1D;
                    }

                    Assert.IsTrue(constrainedAxisIndex > -1);

                    // We expect the prismatic axis in both anchor frames to be parallel and in the same direction.
                    var axisA = new float3x3(anchorAWorld.rot)[constrainedAxisIndex];
                    var axisB = new float3x3(anchorBWorld.rot)[constrainedAxisIndex];
                    var absCosAngle = math.dot(axisA, axisB);
                    if (absCosAngle < OrientationErrorTolCos)
                    {
                        Errors.Add($"Prismatic axis orientation violated by {math.acos(absCosAngle)} radians, which exceeds orientation error tolerance of {OrientationErrorTol} radians",
                            "Prismatic or equivalent", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    // Make sure anchors lie on the prismatic axis:
                    // The anchor position in A lies on the prismatic axis (i.e., axisA) by design since both are attached to the same rigid body A.
                    // So we only need to check that the distance of the anchor position in B to the prismatic axis in A lies below the
                    // provided position error tolerance.
                    var ab = anchorBWorld.pos - anchorAWorld.pos;
                    // calculate rejection of ab with respect to plane formed by axisA and anchorAWorld.pos
                    ab -= math.dot(ab, axisA) * axisA;
                    var distToPrismaticAxisSq = math.lengthsq(ab);
                    if (distToPrismaticAxisSq > PositionErrorTolSq)
                    {
                        Errors.Add($"Joint anchor lies {math.sqrt(distToPrismaticAxisSq)} meters from prismatic axis, which exceeds position error tolerance of {PositionErrorTol} meters.",
                            "Prismatic or equivalent", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    break;
                }
                case JointType.LinearVelocityMotor:
                {
                    var motorConstraint = joint[0];
                    ValidateConstraintType(motorConstraint, ConstraintType.LinearVelocityMotor);
                    var constrainedAxisIndex = motorConstraint.ConstrainedAxis1D;

                    // We expect the linear velocity motor axis (prismatic axis) in both anchor frames to be parallel and in the same direction
                    var axisA = new float3x3(anchorAWorld.rot)[constrainedAxisIndex];
                    var axisB = new float3x3(anchorBWorld.rot)[constrainedAxisIndex];
                    var absCosAngle = math.dot(axisA, axisB);
                    if (absCosAngle < OrientationErrorTolCos)
                    {
                        Errors.Add($"Prismatic axis orientation violated by {math.acos(absCosAngle)} radians, which exceeds orientation error tolerance of {OrientationErrorTol} radians",
                            "LinearVelocityMotor", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    // We also expect the anchor position in A to lie on the prismatic axis attached to B.
                    var ba = anchorAWorld.pos - anchorBWorld.pos;
                    // calculate rejection of ba with respect to plane formed by axisB and anchorBWorld.pos
                    ba -= math.dot(ba, axisB) * axisB;
                    var distToPrismaticAxisSq = math.lengthsq(ba);
                    if (distToPrismaticAxisSq > PositionErrorTolSq)
                    {
                        Errors.Add($"Joint anchor lies {math.sqrt(distToPrismaticAxisSq)} meters from prismatic axis, which exceeds position error tolerance of {PositionErrorTol} meters.",
                            "LinearVelocityMotor", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    break;
                }
                case JointType.LimitedDistance:
                {
                    var distanceConstraint = joint[0];
                    ValidateConstraintType(distanceConstraint, ConstraintType.Linear);

                    var min = distanceConstraint.Min;
                    var max = distanceConstraint.Max;

                    var deltaPos = anchorAWorld.pos - anchorBWorld.pos;
                    var distance = math.length(deltaPos);

                    if (distance < min - PositionErrorTol || distance > max + PositionErrorTol)
                    {
                        Errors.Add($"Joint distance {distance} is out of admissible (min, max) range ({min}, {max}) by more than position error tolerance of {PositionErrorTol} meters.",
                            "LimitedDistance", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    break;
                }
                default:
                    break;
            }

            // target validation for PhysicsJoints
            switch (joint.JointType)
            {
                case JointType.AngularVelocityMotor:
                {
                    // get expected angular velocity
                    var motorConstraint = joint[2];
                    ValidateConstraintType(motorConstraint, ConstraintType.AngularVelocityMotor);
                    int constrainedAxisIndex = motorConstraint.ConstrainedAxis1D;

                    // expected angular velocity in world space
                    var speed = motorConstraint.Target[constrainedAxisIndex];
                    var expectedAngVelRel = new float3x3(anchorAWorld.rot)[constrainedAxisIndex] * speed;

                    // get actual angular velocity
                    var wA = bodyAIsStatic ? float3.zero
                        : PhysicsVelocityLookup[bodyPair.EntityA].GetAngularVelocityWorldSpace(PhysicsMassLookup[bodyPair.EntityA], bodyAWorld.rot);
                    var wB = bodyBIsStatic ? float3.zero
                        : PhysicsVelocityLookup[bodyPair.EntityB].GetAngularVelocityWorldSpace(PhysicsMassLookup[bodyPair.EntityB], bodyBWorld.rot);

                    // actual angular velocity in world space (relative to B)
                    var angVelRel = wA - wB;
                    var check = math.abs(math.lengthsq(expectedAngVelRel - angVelRel));
                    if (check > AngVelErrorTolSq)
                    {
                        Errors.Add($"Angular joint velocity {angVelRel} differs from expected angular velocity {expectedAngVelRel} by {math.sqrt(check)} rad/s which is more than the provided error tolerance of {AngVelErrorTol} rad/s.",
                            "AngularVelocityMotor", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    break;
                }
                case JointType.LinearVelocityMotor:
                {
                    var motorConstraint = joint[0];
                    ValidateConstraintType(motorConstraint, ConstraintType.LinearVelocityMotor);
                    int constrainedAxisIndex = motorConstraint.ConstrainedAxis1D;

                    // expected angular velocity in world space
                    var speed = motorConstraint.Target[constrainedAxisIndex];
                    var expectedLinVelRel = new float3x3(anchorBWorld.rot)[constrainedAxisIndex] * speed;

                    // get actual linear velocity
                    var vA = bodyAIsStatic ? float3.zero : PhysicsVelocityLookup[bodyPair.EntityA].Linear;
                    var vB = bodyBIsStatic ? float3.zero : PhysicsVelocityLookup[bodyPair.EntityB].Linear;

                    // actual linear velocity in world space (relative to B)
                    var linVelRel = vA - vB;

                    if (math.abs(math.lengthsq(expectedLinVelRel - linVelRel)) > LinVelErrorTolSq)
                    {
                        Errors.Add($"Linear joint velocity {linVelRel} differs from expected linear velocity {expectedLinVelRel} by more than provided error tolerance of {LinVelErrorTol} m/s.",
                            "LinearVelocityMotor", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    break;
                }
                case JointType.RotationalMotor:
                {
                    var motorConstraint = joint[0];
                    ValidateConstraintType(motorConstraint, ConstraintType.RotationMotor);
                    int constrainedAxisIndex = motorConstraint.ConstrainedAxis1D;
                    var targetAngle = motorConstraint.Target[constrainedAxisIndex];

                    // Calculate angle between the joint attachment frames.
                    // Note: we already confirmed that the joint axis is aligned in both anchor frames in the pose validation above.
                    var qDelta = math.normalize(math.mul(math.inverse(anchorBWorld.rot), anchorAWorld.rot));
                    ((Quaternion)qDelta).ToAngleAxis(out var currentAngle, out var axis);
                    // account for flip of axis in ToAngleAxis calculation
                    currentAngle *= axis[constrainedAxisIndex];
                    currentAngle = math.radians(currentAngle);
                    var deltaAngle = currentAngle - targetAngle;
                    var deltaAngleCos = math.cos(deltaAngle);
                    // Note: below we exclude compliant joints, since these won't be able to reach their targets with reasonable accuracy in the general case.
                    var compliantJoint = motorConstraint.SpringFrequency < 1e3;
                    if (deltaAngleCos < OrientationErrorTolCos && !compliantJoint)
                    {
                        Errors.Add($"Angle between anchor frames differs from target angle {targetAngle} radians by {deltaAngle} radians, which exceeds the orientation error tolerance of {OrientationErrorTol} radians.",
                            "RotationalMotor", bodyPair.EntityA, bodyPair.EntityB);
                    }

                    // check if we are within the limits
                    if (currentAngle + OrientationErrorTol <= motorConstraint.Min || currentAngle - OrientationErrorTol >= motorConstraint.Max)
                    {
                        Errors.Add($"Angle between anchor frames {currentAngle} is out of admissible (min, max) range ({motorConstraint.Min}, {motorConstraint.Max}) by more than orientation error tolerance of {OrientationErrorTol} radians.",
                            "RotationalMotor", bodyPair.EntityA, bodyPair.EntityB);
                    }
                    break;
                }
                case JointType.PositionalMotor:
                {
                    var motorConstraint = joint[0];
                    ValidateConstraintType(motorConstraint, ConstraintType.PositionMotor);
                    int constrainedAxisIndex = motorConstraint.ConstrainedAxis1D;
                    var targetCoordinate = motorConstraint.Target[constrainedAxisIndex];
                    var prismaticAxis = new float3x3(anchorAWorld.rot)[constrainedAxisIndex];
                    var targetAnchorPosA = prismaticAxis * targetCoordinate + anchorBWorld.pos;
                    var error = math.lengthsq(targetAnchorPosA - anchorAWorld.pos);
                    if (error > PositionErrorTolSq)
                    {
                        Errors.Add($"Joint anchor lies {math.sqrt(error)} meters from target position, which exceeds position error tolerance of {PositionErrorTol} meters.",
                            "PositionalMotor", bodyPair.EntityA, bodyPair.EntityB);
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    [BurstCompile]
    public partial struct ValidateRigidBodyMotionJob : IJobEntity
    {
        [NativeDisableUnsafePtrRestriction] public SimulationValidationSystem.ErrorCounter Errors;

        [ReadOnly] public float MaxLinVel;
        [ReadOnly] public float MaxAngVel;
        [ReadOnly] public float MaxLinVelSq;
        [ReadOnly] public float MaxAngVelSq;
        [ReadOnly] public float3 ExpectedLinVel;
        [ReadOnly] public float3 ExpectedAngVel;

        [GenerateTestsForBurstCompatibility]
        void Execute(Entity entity, in LocalTransform transform, in PhysicsVelocity pv, in PhysicsMass pm)
        {
            var v = pv.Linear;
            var w = pv.GetAngularVelocityWorldSpace(pm, transform.Rotation);
            var vDiffSq = math.lengthsq(v - ExpectedLinVel);
            var wDiffSq = math.lengthsq(w - ExpectedAngVel);
            bool linVelReached = vDiffSq <= MaxLinVelSq;
            bool angVelReached = wDiffSq <= MaxAngVelSq;
            if (!linVelReached || !angVelReached)
            {
                Errors.Add(
                    $"The (linear, angular) velocity ({v}, {w}) exceeds the expected velocity ({ExpectedLinVel}, {ExpectedAngVel}) by more than the (linear, angular) velocity error tolerance of ({MaxLinVel}, {MaxAngVel}).",
                    "Rigid Body", entity);
            }
        }
    }

    [UpdateInGroup(typeof(AfterPhysicsSystemGroup))]
    public partial struct SimulationValidationSystem : ISystem
    {
        private ComponentLookup<LocalTransform> TransformLookup;
        private ComponentLookup<LocalToWorld> LocalToWorldLookup;

        private ComponentLookup<PhysicsVelocity> PhysicsVelocityLookup;
        private ComponentLookup<PhysicsMass> PhysicsMassLookup;
        private int NumErrorsDetected;
        private ErrorCounter Errors;

        private float ElapsedTime;

        public struct ErrorCounter
        {
            private UnsafeAtomicCounter32 Counter;

            public unsafe ErrorCounter(int* errorCount)
            {
                Counter = new UnsafeAtomicCounter32(errorCount);
            }

            public void Add(in FixedString4096Bytes errorMessage)
            {
                Debug.LogWarning($"{errorMessage}");
                Counter.Add(1);
            }

            public void Add(in FixedString4096Bytes errorMessage, in FixedString128Bytes errorCategory, Entity entity)
            {
                Add($"Validation ({errorCategory}, Entity: {entity.ToFixedString()}): {errorMessage}");
            }

            public void Add(in FixedString4096Bytes errorMessage, in FixedString128Bytes errorCategory, Entity entityA, Entity entityB)
            {
                Add($"Validation ({errorCategory}, Entities: {entityA.ToFixedString()}, {entityB.ToFixedString()}): {errorMessage}");
            }

            public unsafe int GetCount()
            {
                return *Counter.Counter;
            }

            public void Reset()
            {
                Counter.Reset();
            }
        }

        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<SimulationValidationSettings>();
            state.RequireForUpdate<PhysicsWorldSingleton>();

            TransformLookup = state.GetComponentLookup<LocalTransform>();
            LocalToWorldLookup = state.GetComponentLookup<LocalToWorld>();

            PhysicsVelocityLookup = state.GetComponentLookup<PhysicsVelocity>();
            PhysicsMassLookup = state.GetComponentLookup<PhysicsMass>();
            unsafe
            {
                fixed(int* numErrorsDetectedPtr = &NumErrorsDetected)
                {
                    Errors = new ErrorCounter(numErrorsDetectedPtr);
                }
            }
            ElapsedTime = 0.0f;
        }

        public void OnDestroy(ref SystemState state)
        {
        }

        public void OnUpdate(ref SystemState state)
        {
            // since we require SimulationValidationSettings to exist for this system to be updated,
            // we can be sure that we can retrieve it.
            var settings = SystemAPI.GetSingleton<SimulationValidationSettings>();

            if (!settings.EnableValidation)
            {
                return;
            }
            // else:

            // check if any error has been detected in the validation jobs scheduled last frame (see below)
            var numErrorsDetectedLastFrame = Errors.GetCount();
            // reset the error counter for the upcoming validation jobs
            Errors.Reset();

            // Note: we need to calculate our own elapsed time since the first update of this system has occurred. This is because during SubScene streaming with closed SubScenes
            // the systems in the SubScenes are not immediately created and stepped. They might get stepped only after a few frames delay. Therefore, some time might already have
            // passed (i.e., SystemAPI.Time.ElapsedTime > 0) the first time this system is updated.
            var elapsedTime = ElapsedTime;
            ElapsedTime += SystemAPI.Time.DeltaTime;
            if (settings.ValidationTimeRange[0] <= elapsedTime && (elapsedTime <= settings.ValidationTimeRange[1] || settings.ValidationTimeRange[1] < 0))
            {
                TransformLookup.Update(ref state);
                LocalToWorldLookup.Update(ref state);

                PhysicsVelocityLookup.Update(ref state);
                PhysicsMassLookup.Update(ref state);

                var physicsWorld = SystemAPI.GetSingleton<PhysicsWorldSingleton>();
                var combinedHandle = new JobHandle();

                if (settings.ValidateJointBehavior)
                {
                    var handle = new ValidateJointBehaviorJob()
                    {
                        Errors = Errors,
                        TransformLookup = TransformLookup,
                        LocalToWorldLookup = LocalToWorldLookup,

                        PhysicsVelocityLookup = PhysicsVelocityLookup,
                        PhysicsMassLookup = PhysicsMassLookup,
                        DynamicsWorld = physicsWorld.DynamicsWorld,
                        Joints = physicsWorld.DynamicsWorld.Joints,

                        PositionErrorTol = settings.PositionErrorTolerance,
                        PositionErrorTolSq = settings.PositionErrorTolerance * settings.PositionErrorTolerance,
                        OrientationErrorTol = settings.OrientationErrorTolerance,
                        OrientationErrorTolCos = math.cos(settings.OrientationErrorTolerance),
                        AngVelErrorTol = settings.AngularVelocityErrorTolerance,
                        AngVelErrorTolSq = settings.AngularVelocityErrorTolerance * settings.AngularVelocityErrorTolerance,
                        LinVelErrorTol = settings.LinearVelocityErrorTolerance,
                        LinVelErrorTolSq = settings.LinearVelocityErrorTolerance * settings.LinearVelocityErrorTolerance,
                    }.ScheduleParallel(state.Dependency);
                    combinedHandle = JobHandle.CombineDependencies(combinedHandle, handle);
                }

                if (settings.ValidateRigidBodyMotion)
                {
                    var handle = new ValidateRigidBodyMotionJob()
                    {
                        Errors = Errors,
                        MaxLinVel = settings.LinearVelocityErrorTolerance,
                        MaxAngVel = settings.AngularVelocityErrorTolerance,
                        MaxLinVelSq = settings.LinearVelocityErrorTolerance * settings.LinearVelocityErrorTolerance,
                        MaxAngVelSq = settings.AngularVelocityErrorTolerance * settings.AngularVelocityErrorTolerance,
                        ExpectedLinVel = settings.ExpectedLinearVelocity,
                        ExpectedAngVel = settings.ExpectedAngularVelocity
                    }.ScheduleParallel(state.Dependency);
                    combinedHandle = JobHandle.CombineDependencies(combinedHandle, handle);
                }

                state.Dependency = combinedHandle;
            }

            // Assert if last frame any errors have been detected
            Assert.AreEqual(0, numErrorsDetectedLastFrame, $"SimulationValidationSystem: {numErrorsDetectedLastFrame} errors detected in simulation.");
        }
    }

    public struct KineticEnergy : IComponentData
    {
        public float Linear;
        public float Angular;
    }

    /// <summary>
    /// Computes the kinetic energy of individual rigid bodies before the physics step.
    /// </summary>
    [UpdateInGroup(typeof(BeforePhysicsSystemGroup))]
    [RequireMatchingQueriesForUpdate]
    public partial struct ComputeKineticEnergySystem : ISystem
    {
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<SimulationValidationSettings>();
            state.RequireForUpdate<PhysicsWorldSingleton>();
        }

        public void OnUpdate(ref SystemState state)
        {
            var settings = SystemAPI.GetSingleton<SimulationValidationSettings>();
            if (!settings.EnableValidation || !settings.ValidateKineticEnergyConservation)
            {
                return;
            }

            // Add a KineticEnergy component to all dynamic bodies which don't have one yet
            var ecb = new EntityCommandBuffer(Allocator.Temp);
            foreach (var(mass, velocity, entity)
                     in SystemAPI.Query<PhysicsMass, PhysicsVelocity>().WithEntityAccess()
                         .WithNone<KineticEnergy>())
            {
                ecb.AddComponent<KineticEnergy>(entity);
            }

            ecb.Playback(state.EntityManager);

            // Compute kinetic energy for all dynamic bodies with a KineticEnergyData component using idiomatic for loop
            foreach (var(velocity, mass, kineticEnergy)
                     in SystemAPI.Query<RefRO<PhysicsVelocity>, RefRO<PhysicsMass>, RefRW<KineticEnergy>>())
            {
                kineticEnergy.ValueRW.Linear = velocity.ValueRO.GetLinearKineticEnergy(mass.ValueRO);
                kineticEnergy.ValueRW.Angular = velocity.ValueRO.GetAngularKineticEnergy(mass.ValueRO);
            }
        }
    }

    /// <summary>
    /// Checks if kinetic energy for individual rigid bodies has been conserved during the last physics step.
    /// </summary>
    [UpdateInGroup(typeof(AfterPhysicsSystemGroup))]
    [RequireMatchingQueriesForUpdate]
    public partial struct ValidateKinematicEnergyConservationSystem : ISystem
    {
        private float ElapsedTime;

        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<SimulationValidationSettings>();
            state.RequireForUpdate<PhysicsWorldSingleton>();
            ElapsedTime = 0.0f;
        }

        public void OnUpdate(ref SystemState state)
        {
            var settings = SystemAPI.GetSingleton<SimulationValidationSettings>();
            if (!settings.EnableValidation || !settings.ValidateKineticEnergyConservation)
            {
                return;
            }

            var elapsedTime = ElapsedTime;
            ElapsedTime += SystemAPI.Time.DeltaTime;
            if (settings.ValidationTimeRange[0] <= elapsedTime && (elapsedTime <= settings.ValidationTimeRange[1] || settings.ValidationTimeRange[1] < 0))
            {
                // Compute kinetic energy for all dynamic bodies after the integration step and check if it is conserved
                foreach (var(velocity, mass, gravityFactor, kineticEnergy)
                         in SystemAPI.Query<RefRO<PhysicsVelocity>, RefRO<PhysicsMass>, RefRO<PhysicsGravityFactor>, RefRO<KineticEnergy>>())
                {
                    // Rigid bodies that are affected by gravity are excluded from this validation
                    if (gravityFactor.ValueRO.Value == 0)
                    {
                        var linearKineticEnergy = velocity.ValueRO.GetLinearKineticEnergy(mass.ValueRO);
                        var angularKineticEnergy = velocity.ValueRO.GetAngularKineticEnergy(mass.ValueRO);
                        Assert.AreApproximatelyEqual(kineticEnergy.ValueRO.Linear, linearKineticEnergy,
                            settings.LinearKineticEnergyTolerance);
                        Assert.AreApproximatelyEqual(kineticEnergy.ValueRO.Angular, angularKineticEnergy,
                            settings.AngularKineticEnergyTolerance);
                    }
                }
            }
        }
    }
}
