using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// INTERNAL UnityPhysics interface for jobs that iterate through the list of contact manifolds
    /// produced by the narrow phase Important: Only use inside UnityPhysics code! Jobs in other
    /// projects should implement IContactsJob.
    /// </summary>
    [JobProducerType(typeof(IContactsJobExtensions.ContactsJobProcess<>))]
    public interface IContactsJobBase
    {
        /// <summary>
        /// Execute an operation on given header and contact.
        /// Note, multiple contacts can share the same header, but will have a different
        /// ModifiableContactPoint.Index.
        /// </summary>
        ///
        /// <param name="header">   [in,out] The header. </param>
        /// <param name="contact">  [in,out] The contact. </param>
        void Execute(ref ModifiableContactHeader header, ref ModifiableContactPoint contact);
    }


    /// <summary>
    /// Interface for jobs that iterate through the list of contact manifolds produced by the narrow
    /// phase.
    /// </summary>
    public interface IContactsJob : IContactsJobBase
    {
    }


    /// <summary>   A modifiable contact header. </summary>
    public struct ModifiableContactHeader
    {
        internal ContactHeader ContactHeader;

        /// <summary>   Gets a value indicating whether this object is modified. </summary>
        ///
        /// <value> True if modified, false if not. </value>
        public bool Modified { get; private set; }

        internal EntityPair EntityPair;

        /// <summary>   Gets the entity b. </summary>
        ///
        /// <value> The entity b. </value>
        public Entity EntityB => EntityPair.EntityB;

        /// <summary>   Gets the entity a. </summary>
        ///
        /// <value> The entity a. </value>
        public Entity EntityA => EntityPair.EntityA;

        /// <summary>   Gets the body index b. </summary>
        ///
        /// <value> The body index b. </value>
        public int BodyIndexB => ContactHeader.BodyPair.BodyIndexB;

        /// <summary>   Gets the body index a. </summary>
        ///
        /// <value> The body index a. </value>
        public int BodyIndexA => ContactHeader.BodyPair.BodyIndexA;

        /// <summary>   Gets the custom tags b. </summary>
        ///
        /// <value> The custom tags b. </value>
        public byte CustomTagsB => ContactHeader.BodyCustomTags.CustomTagsB;

        /// <summary>   Gets the custom tags a. </summary>
        ///
        /// <value> The custom tags a. </value>
        public byte CustomTagsA => ContactHeader.BodyCustomTags.CustomTagsA;

        /// <summary>   Gets the collider key b. </summary>
        ///
        /// <value> The collider key b. </value>
        public ColliderKey ColliderKeyB => ContactHeader.ColliderKeys.ColliderKeyB;

        /// <summary>   Gets the collider key a. </summary>
        ///
        /// <value> The collider key a. </value>
        public ColliderKey ColliderKeyA => ContactHeader.ColliderKeys.ColliderKeyA;

        /// <summary>   Gets the number of contacts. </summary>
        ///
        /// <value> The total number of contacts. </value>
        public int NumContacts => ContactHeader.NumContacts;

        /// <summary>   Gets or sets the jacobian flags. </summary>
        ///
        /// <value> Options that control the jacobian. </value>
        public JacobianFlags JacobianFlags
        {
            get => ContactHeader.JacobianFlags;
            set
            {
                ContactHeader.JacobianFlags = value;
                Modified = true;
            }
        }

        /// <summary>   Gets or sets the normal. </summary>
        ///
        /// <value> The normal. </value>
        public float3 Normal
        {
            get => ContactHeader.Normal;
            set
            {
                ContactHeader.Normal = value;
                Modified = true;
            }
        }

        /// <summary>   Gets or sets the coefficient of friction. </summary>
        ///
        /// <value> The coefficient of friction. </value>
        public float CoefficientOfFriction
        {
            get => ContactHeader.CoefficientOfFriction;
            set
            {
                ContactHeader.CoefficientOfFriction = value;
                Modified = true;
            }
        }

        /// <summary>   Gets or sets the coefficient of restitution. </summary>
        ///
        /// <value> The coefficient of restitution. </value>
        public float CoefficientOfRestitution
        {
            get => ContactHeader.CoefficientOfRestitution;
            set
            {
                ContactHeader.CoefficientOfRestitution = value;
                Modified = true;
            }
        }
    }

    /// <summary>   A modifiable contact point. </summary>
    public struct ModifiableContactPoint
    {
        internal ContactPoint ContactPoint;

        /// <summary>   Gets a value indicating whether this object is modified. </summary>
        ///
        /// <value> True if modified, false if not. </value>
        public bool Modified { get; private set; }

        /// <summary>   Index of this point, within the ModifiableContactHeader. </summary>
        ///
        /// <value> The index. </value>
        public int Index { get; internal set; }

        /// <summary>   Gets or sets the position. </summary>
        ///
        /// <value> The position. </value>
        public float3 Position
        {
            get => ContactPoint.Position;
            set
            {
                ContactPoint.Position = value;
                Modified = true;
            }
        }

        /// <summary>   Gets or sets the distance. </summary>
        ///
        /// <value> The distance. </value>
        public float Distance
        {
            get => ContactPoint.Distance;
            set
            {
                ContactPoint.Distance = value;
                Modified = true;
            }
        }
    }

    /// <summary>   The contacts job extensions. </summary>
    public static class IContactsJobExtensions
    {
        /// <summary>   Schedules an IContactsJob for serial processing. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="job">      The scheduled job. </param>
        /// <param name="simulationSingleton">  The simulation singleton. </param>
        /// <param name="world">                [in,out] The physics world. </param>
        /// <param name="inputDeps">            The input dependencies. </param>
        ///
        /// <returns>   A JobHandle. </returns>
        public static JobHandle Schedule<T>(this T job, SimulationSingleton simulationSingleton, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IContactsJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }

            return ScheduleUnityPhysicsContactsJob(job, simulationSingleton.AsSimulation(), ref world, inputDeps);
        }

        /// <summary>   Schedules an IContactsJob for parallel processing. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="job">      The scheduled job. </param>
        /// <param name="innerLoopBatchCount">  Granularity in which workstealing is performed. A value of N, means the
        ///                                     job queue will combine N job executions and perform them in an efficient
        ///                                     inner loop.</param>
        /// <param name="simulationSingleton">  The simulation singleton. </param>
        /// <param name="world">                [in,out] The physics world. </param>
        /// <param name="inputDeps">            The input dependencies. </param>
        ///
        /// <returns>   A JobHandle. </returns>
        public static JobHandle ScheduleParallel<T>(this T job, int innerLoopBatchCount, SimulationSingleton simulationSingleton, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IContactsJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }

            return ScheduleParallelUnityPhysicsContactsJob(job, innerLoopBatchCount, simulationSingleton.AsSimulation(), ref world, inputDeps);
        }

        static unsafe JobHandle ScheduleUnityPhysicsContactsJob<T>(this T job, Simulation simulation, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IContactsJobBase
        {
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.PostCreateContacts);

            if (simulation.StepContext.Contacts.IsCreated)
            {
                var data = new ContactsJobData<T>
                {
                    UserJobData = job,
                    ContactReader = simulation.StepContext.Contacts.AsReader(),
                    Bodies = world.Bodies,
                    IsParallel = false
                };

                var reflectionData = ContactsJobProcess<T>.jobReflectionData.Data;
                ContactsJobProcess<T>.CheckReflectionDataCorrect(reflectionData);

                var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), reflectionData, inputDeps, ScheduleMode.Single);
                return JobsUtility.Schedule(ref parameters);
            }

            return inputDeps;
        }

        static unsafe JobHandle ScheduleParallelUnityPhysicsContactsJob<T>(this T job, int innerLoopBatchCount, Simulation simulation, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IContactsJobBase
        {
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.PostCreateContacts);

            if (simulation.StepContext.Contacts.IsCreated)
            {
                var contactsStream = simulation.StepContext.Contacts;
                var data = new ContactsJobData<T>
                {
                    UserJobData = job,
                    ContactReader = contactsStream.AsReader(),
                    Bodies = world.Bodies,
                    IsParallel = true
                };

                var reflectionData = ContactsJobProcess<T>.jobReflectionData.Data;
                ContactsJobProcess<T>.CheckReflectionDataCorrect(reflectionData);

                var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), reflectionData, inputDeps, ScheduleMode.Parallel);
                var forEachCountPtr = NativeStreamUnsafeUtility.GetUnsafeForEachCountPtr(ref contactsStream);
                var listDataPtr = (byte*)forEachCountPtr - sizeof(void*);
                return JobsUtility.ScheduleParallelForDeferArraySize(ref parameters, innerLoopBatchCount, listDataPtr, null);
            }

            return inputDeps;
        }

        internal struct ContactsJobData<T> where T : struct
        {
            public T UserJobData;

            [NativeDisableContainerSafetyRestriction] public NativeStream.Reader ContactReader;
            // Disable aliasing restriction in case T has a NativeArray of PhysicsWorld.Bodies
            [ReadOnly, NativeDisableContainerSafetyRestriction] public NativeArray<RigidBody> Bodies;
            public bool IsParallel;
        }

        internal struct ContactsJobProcess<T> where T : struct, IContactsJobBase
        {
            internal static readonly SharedStatic<IntPtr> jobReflectionData = SharedStatic<IntPtr>.GetOrCreate<ContactsJobProcess<T>>();

            [Preserve]
            public static void Initialize()
            {
                if (jobReflectionData.Data == IntPtr.Zero)
                    jobReflectionData.Data = JobsUtility.CreateJobReflectionData(typeof(ContactsJobData<T>), typeof(T), (ExecuteJobFunction)Execute);
            }

            public delegate void ExecuteJobFunction(ref ContactsJobData<T> jobData, IntPtr additionalData,
                IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex);

            public unsafe static void Execute(ref ContactsJobData<T> jobData, IntPtr additionalData,
                IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex)
            {
                while (true)
                {
                    int forEachIndexBegin = 0;
                    int forEachIndexEnd = jobData.ContactReader.ForEachCount;

                    if (jobData.IsParallel)
                    {
                        if (!JobsUtility.GetWorkStealingRange(ref ranges, jobIndex, out forEachIndexBegin, out forEachIndexEnd))
                            break;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                        JobsUtility.PatchBufferMinMaxRanges(bufferRangePatchData, UnsafeUtility.AddressOf(ref jobData), forEachIndexBegin, forEachIndexEnd - forEachIndexBegin);
#endif
                    }

                    var iterator = new ContactsJobIterator(jobData.ContactReader, forEachIndexBegin, forEachIndexEnd);

                    while (iterator.HasItemsLeft())
                    {
                        iterator.Next();

                        //<todo.eoin.modifier Could store the pointer, to avoid copies, like the jacobian job?
                        var header = new ModifiableContactHeader
                        {
                            ContactHeader = *iterator.m_LastHeader,
                            EntityPair = new EntityPair
                            {
                                EntityA = jobData.Bodies[iterator.m_LastHeader->BodyPair.BodyIndexA].Entity,
                                EntityB = jobData.Bodies[iterator.m_LastHeader->BodyPair.BodyIndexB].Entity
                            }
                        };
                        var contact = new ModifiableContactPoint
                        {
                            ContactPoint = *iterator.m_LastContact,
                            Index = iterator.CurrentPointIndex
                        };

                        jobData.UserJobData.Execute(ref header, ref contact);

                        if (header.Modified)
                        {
                            *iterator.m_LastHeader = header.ContactHeader;
                        }

                        if (contact.Modified)
                        {
                            *iterator.m_LastContact = contact.ContactPoint;
                        }
                    }

                    // If we are not running in parallel, we are done.
                    if (!jobData.IsParallel)
                        break;
                }
            }

            [System.Diagnostics.Conditional("ENABLE_UNITY_COLLECTIONS_CHECK")]
            internal static void CheckReflectionDataCorrect(IntPtr reflectionData)
            {
                if (reflectionData == IntPtr.Zero)
                    SafetyChecks.ThrowInvalidOperationException("Reflection data was not set up by an Initialize() call");
            }
        }

        /// <summary>   Early job initialize. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        public static void EarlyJobInit<T>()
            where T : struct, IContactsJobBase
        {
            ContactsJobProcess<T>.Initialize();
        }

        // Utility to help iterate over all the items in the contacts job stream
        unsafe struct ContactsJobIterator
        {
            [NativeDisableContainerSafetyRestriction] NativeStream.Reader m_ContactReader;
            [NativeDisableUnsafePtrRestriction] public ContactHeader* m_LastHeader;
            [NativeDisableUnsafePtrRestriction] public ContactPoint* m_LastContact;
            int m_NumPointsLeft;
            int m_CurrentForEachIndex;
            int m_ForEachIndexEnd;

            public ContactsJobIterator(NativeStream.Reader reader, int forEachIndexBegin, int forEachIndexEnd)
            {
                SafetyChecks.CheckAreEqualAndThrow(true, forEachIndexBegin >= 0
                    && forEachIndexBegin <= forEachIndexEnd // Note: we use <= here since for empty readers,
                                                            // forEachIndexEnd will be identical to forEachIndexBegin,
                                                            // both being zero. This is still valid, and should not throw.
                    && forEachIndexEnd <= reader.ForEachCount);

                m_ContactReader = reader;

                m_CurrentForEachIndex = forEachIndexBegin;
                m_ForEachIndexEnd = forEachIndexEnd;
                m_NumPointsLeft = 0;
                m_LastHeader = null;
                m_LastContact = null;

                AdvanceForEachIndex();
            }

            public bool HasItemsLeft()
            {
                return m_ContactReader.RemainingItemCount > 0;
            }

            public int CurrentPointIndex => m_LastHeader->NumContacts - m_NumPointsLeft - 1;

            public void Next()
            {
                if (HasItemsLeft())
                {
                    if (m_NumPointsLeft == 0)
                    {
                        // Need to get a new header
                        m_LastHeader = (ContactHeader*)m_ContactReader.ReadUnsafePtr(sizeof(ContactHeader));
                        m_NumPointsLeft = m_LastHeader->NumContacts;
                        AdvanceForEachIndex();
                    }

                    m_LastContact = (ContactPoint*)m_ContactReader.ReadUnsafePtr(sizeof(ContactPoint));
                    m_NumPointsLeft--;
                    AdvanceForEachIndex();
                }
            }

            void AdvanceForEachIndex()
            {
                while (m_ContactReader.RemainingItemCount == 0 && m_CurrentForEachIndex < m_ForEachIndexEnd)
                {
                    m_ContactReader.BeginForEachIndex(m_CurrentForEachIndex++);
                }
            }
        }
    }
}
