using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// An event raised when a voxel block pair collision is detected during narrowphase.
    /// Contains detailed voxel-level contact information including block coordinates.
    /// </summary>
    public struct VoxelContactEvent // : ISimulationEvent<VoxelContactEvent>
    {
        internal VoxelContactEventData EventData;

        // #region ISimulationEvent
        // public Entity EntityA { get; }
        // public Entity EntityB { get; }
        //
        // public ColliderKey ColliderKeyA { get; }
        // public ColliderKey ColliderKeyB { get; }
        // #endregion

        /// <summary>   Gets the body index A. </summary>
        ///
        /// <value> The body index A. </value>
        public int BodyIndexA => EventData.BodyIndices.BodyIndexA;

        /// <summary>   Gets the body index B. </summary>
        ///
        /// <value> The body index B. </value>
        public int BodyIndexB => EventData.BodyIndices.BodyIndexB;

        /// <summary>   Gets the voxel block coordinates in body A's local space. </summary>
        ///
        /// <value> The voxel coordinates in A. </value>
        public int3 VoxelCoordsInA => EventData.VoxelCoordsInA;

        /// <summary>   Gets the voxel block coordinates in body B's local space. </summary>
        ///
        /// <value> The voxel coordinates in B. </value>
        public int3 VoxelCoordsInB => EventData.VoxelCoordsInB;

        /// <summary>   Gets the contact normal (from B to A, world space). </summary>
        ///
        /// <value> The normal vector. </value>
        public float3 Normal => EventData.Normal;

        /// <summary>   Gets the contact metadata (flags). </summary>
        ///
        /// <value> True if the contact affects the physics simulation. </value>
        public bool IsPhysicsContact => EventData.isPhysicsContact;

        /// <summary>   Gets the separating distance along the normal (negative = penetrating). </summary>
        public float Distance => EventData.Distance;

        /// <summary>   True if this contact was fused into a bilateral (equality) constraint. </summary>
        public bool IsBilateral => (EventData.DebugFlags & VoxelContactEventData.FlagBilateral) != 0;

        /// <summary>   True if this contact uses a diagonal (rounded corner/edge) normal, not a grid axis. </summary>
        public bool IsDiagonal => (EventData.DebugFlags & VoxelContactEventData.FlagDiagonal) != 0;

        /// <summary>
        /// Compares this VoxelContactEvent object to another to determine their relative ordering.
        /// </summary>
        ///
        /// <param name="other">    Another instance to compare. </param>
        ///
        /// <returns>
        /// Negative if this object is less than the other, 0 if they are equal, or positive if this is
        /// greater.
        /// </returns>
        // public int CompareTo(VoxelContactEvent other) => ISimulationEventUtilities.CompareEvents(this, other);
    }

    /// <summary>
    /// A stream of voxel contact events. This is a value type, which means it can be used in Burst jobs
    /// (unlike IEnumerable&lt;VoxelContactEvent&gt;).
    /// </summary>
    public struct VoxelContactEvents /* : IEnumerable<VoxelContactEvent> */
    {
        [NativeDisableContainerSafetyRestriction]
        private readonly NativeStream m_EventDataStream;

        internal VoxelContactEvents(NativeStream eventDataStream)
        {
            m_EventDataStream = eventDataStream;
        }

        /// <summary>   Gets the enumerator. </summary>
        ///
        /// <returns>   The enumerator. </returns>
        public Enumerator GetEnumerator()
        {
            return new Enumerator(m_EventDataStream);
        }

        /// <summary>   An enumerator. </summary>
        public struct Enumerator /* : IEnumerator<VoxelContactEvent> */
        {
            private NativeStream.Reader m_Reader;
            private int m_CurrentWorkItem;
            private readonly int m_NumWorkItems;

            /// <summary>   Gets the current voxel contact event. </summary>
            ///
            /// <value> The current event. </value>
            public VoxelContactEvent Current { get; private set; }

            internal Enumerator(NativeStream stream)
            {
                m_Reader = stream.IsCreated ? stream.AsReader() : new NativeStream.Reader();
                m_CurrentWorkItem = 0;
                m_NumWorkItems = stream.IsCreated ? stream.ForEachCount : 0;
                Current = default;

                AdvanceReader();
            }

            /// <summary>   Determines if we can move next. </summary>
            ///
            /// <returns>   True if it succeeds, false if it fails. </returns>
            public bool MoveNext()
            {
                if (m_Reader.RemainingItemCount > 0)
                {
                    var eventData = m_Reader.Read<VoxelContactEventData>();

                    Current = new VoxelContactEvent
                    {
                        EventData = eventData
                    };

                    AdvanceReader();
                    return true;
                }
                return false;
            }

            private void AdvanceReader()
            {
                while (m_Reader.RemainingItemCount == 0 && m_CurrentWorkItem < m_NumWorkItems)
                {
                    m_Reader.BeginForEachIndex(m_CurrentWorkItem);
                    m_CurrentWorkItem++;
                }
            }
        }
    }

    /// <summary>
    /// Internal data structure for voxel contact events.
    /// Compact fixed-size struct optimized for stream storage.
    /// </summary>
    internal struct VoxelContactEventData
    {
        public const byte FlagBilateral = 1 << 0;
        public const byte FlagDiagonal = 1 << 1;

        public BodyIndexPair BodyIndices;
        public int3 VoxelCoordsInA;
        public int3 VoxelCoordsInB;
        public float3 Normal;
        public float Distance;        // separating distance along Normal (negative = penetrating)
        public byte DebugFlags;       // FlagBilateral | FlagDiagonal
        public bool isPhysicsContact; // May change to a bitmask flag later (byte).
    }
}
