namespace Unity.Physics.Authoring
{
    // CAELIX FORK NOTE:
    // Upstream declares this enum in Components/RigidbodyAuthoring.cs, which this fork does not
    // ship (the authoring/baking part of Unity.Physics.Hybrid is dropped - see commit 5739e6f
    // "Remove Editor and Hybrid assemblies, keep core runtime only"). DisplayCollidersSystem
    // needs it to colour bodies by motion type, so it is reproduced verbatim here.
    // If RigidbodyAuthoring.cs is ever restored, DELETE this file - it will collide.

    /// <summary>
    /// Describes how a rigid body will be simulated in the run-time.
    /// </summary>
    public enum BodyMotionType
    {
        /// <summary>
        /// The physics solver will move the rigid body and handle its collision response with other bodies, based on its physical properties.
        /// </summary>
        Dynamic,
        /// <summary>
        /// The physics solver will move the rigid body according to its velocity, but it will be treated as though it has infinite mass.
        /// It will generate a collision response with any rigid bodies that lie in its path of motion, but will not be affected by them.
        /// </summary>
        Kinematic,
        /// <summary>
        /// The physics solver will not move the rigid body.
        /// Any transformations applied to it will be treated as though it is teleporting.
        /// </summary>
        Static
    }
}
