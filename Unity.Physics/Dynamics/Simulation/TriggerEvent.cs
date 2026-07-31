using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;

namespace Unity.Physics
{
    /// <summary>
    /// An event raised when a pair of bodies involving a trigger material have overlapped during
    /// solving.
    /// </summary>
    public struct TriggerEvent : ISimulationEvent<TriggerEvent>
    {
        internal TriggerEventData EventData;

        /// <summary>   Gets the entity b. </summary>
        ///
        /// <value> The entity b. </value>
        public Entity EntityB => EventData.Entities.EntityB;

        /// <summary>   Gets the entity a. </summary>
        ///
        /// <value> The entity a. </value>
        public Entity EntityA => EventData.Entities.EntityA;

        /// <summary>   Gets the body index b. </summary>
        ///
        /// <value> The body index b. </value>
        public int BodyIndexB => EventData.BodyIndices.BodyIndexB;

        /// <summary>   Gets the body index a. </summary>
        ///
        /// <value> The body index a. </value>
        public int BodyIndexA => EventData.BodyIndices.BodyIndexA;

        /// <summary>   Gets the collider key b. </summary>
        ///
        /// <value> The collider key b. </value>
        public ColliderKey ColliderKeyB => EventData.ColliderKeys.ColliderKeyB;

        /// <summary>   Gets the collider key a. </summary>
        ///
        /// <value> The collider key a. </value>
        public ColliderKey ColliderKeyA => EventData.ColliderKeys.ColliderKeyA;

        /// <summary>
        /// Compares this TriggerEvent object to another to determine their relative ordering.
        /// </summary>
        ///
        /// <param name="other">    Another instance to compare. </param>
        ///
        /// <returns>
        /// Negative if this object is less than the other, 0 if they are equal, or positive if this is
        /// greater.
        /// </returns>
        public int CompareTo(TriggerEvent other) => ISimulationEventUtilities.CompareEvents(this, other);
    }

    /// <summary>
    /// A stream of trigger events. This is a value type, which means it can be used in Burst jobs
    /// (unlike IEnumerable&lt;TriggerEvent&gt;).
    /// </summary>
    public struct TriggerEvents /* : IEnumerable<TriggerEvent> */
    {
        [NativeDisableContainerSafetyRestriction]
        readonly NativeStream m_EventDataStream;

        internal TriggerEvents(NativeStream eventDataStream)
        {
            m_EventDataStream = eventDataStream;
        }

        internal NativeStream EventDataStream => m_EventDataStream;

        /// <summary>   Gets the enumerator. </summary>
        ///
        /// <returns>   The enumerator. </returns>
        public Enumerator GetEnumerator()
        {
            return new Enumerator(m_EventDataStream);
        }

        /// <summary>   An enumerator. </summary>
        public struct Enumerator /* : IEnumerator<TriggerEvent> */
        {
            NativeStream.Reader m_Reader;
            int m_CurrentWorkItem;
            readonly int m_WorkItemEnd;

            /// <summary>   Gets or sets the current. </summary>
            ///
            /// <value> The current. </value>
            public TriggerEvent Current { get; private set; }

            internal Enumerator(NativeStream stream)
            {
                m_Reader = stream.IsCreated ? stream.AsReader() : new NativeStream.Reader();
                m_CurrentWorkItem = 0;
                m_WorkItemEnd = stream.IsCreated ? stream.ForEachCount : 0;
                Current = default;

                AdvanceReader();
            }

            internal Enumerator(NativeStream.Reader reader, int forEachIndexBegin, int forEachIndexEnd)
            {
                m_Reader = reader;
                m_CurrentWorkItem = forEachIndexBegin;
                m_WorkItemEnd = forEachIndexEnd;
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
                    var eventData = m_Reader.Read<TriggerEventData>();

                    Current = eventData.CreateTriggerEvent();

                    AdvanceReader();
                    return true;
                }
                return false;
            }

            void AdvanceReader()
            {
                while (m_Reader.RemainingItemCount == 0 && m_CurrentWorkItem < m_WorkItemEnd)
                {
                    m_Reader.BeginForEachIndex(m_CurrentWorkItem);
                    m_CurrentWorkItem++;
                }
            }
        }
    }

    // An event raised when a pair of bodies involving a trigger material have overlapped during solving.
    struct TriggerEventData
    {
        public BodyIndexPair BodyIndices;
        public ColliderKeyPair ColliderKeys;
        public EntityPair Entities;

        internal TriggerEvent CreateTriggerEvent()
        {
            return new TriggerEvent
            {
                EventData = this
            };
        }
    }
}
