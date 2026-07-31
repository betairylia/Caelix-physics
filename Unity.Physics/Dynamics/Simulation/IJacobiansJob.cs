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
    /// INTERNAL UnityPhysics interface for jobs that iterate through the list of Jacobians before
    /// they are solved Important: Only use inside UnityPhysics code! Jobs in other projects should
    /// implement IJacobiansJob.
    /// </summary>
    [JobProducerType(typeof(IJacobiansJobExtensions.JacobiansJobProcess<>))]
    public interface IJacobiansJobBase
    {
        /// <summary>
        /// Executes an operation on a header and a contact jacobian.
        /// Note, multiple Jacobians can share the same header.
        /// </summary>
        ///
        /// <param name="header">   [in,out] The header. </param>
        /// <param name="jacobian"> [in,out] The jacobian. </param>
        void Execute(ref ModifiableJacobianHeader header, ref ModifiableContactJacobian jacobian);

        /// <summary>   Executes an operation on a header and a trigger jacobian. </summary>
        ///
        /// <param name="header">   [in,out] The header. </param>
        /// <param name="jacobian"> [in,out] The jacobian. </param>
        void Execute(ref ModifiableJacobianHeader header, ref ModifiableTriggerJacobian jacobian);
    }


    /// <summary>
    /// Interface for jobs that iterate through the list of Jacobians before they are solved.
    /// </summary>
    public interface IJacobiansJob : IJacobiansJobBase
    {
    }


    /// <summary>   A modifiable jacobian header. </summary>
    public unsafe struct ModifiableJacobianHeader
    {
        internal JacobianHeader* m_Header;

        /// <summary>   Gets a value indicating whether the modifiers was changed. </summary>
        ///
        /// <value> True if modifiers changed, false if not. </value>

        public bool ModifiersChanged { get; private set; }

        /// <summary>   Gets a value indicating whether the angular was changed. </summary>
        ///
        /// <value> True if angular changed, false if not. </value>

        public bool AngularChanged { get; private set; }

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

        public int BodyIndexB => m_Header->BodyPair.BodyIndexB;

        /// <summary>   Gets the body index a. </summary>
        ///
        /// <value> The body index a. </value>

        public int BodyIndexA => m_Header->BodyPair.BodyIndexA;

        /// <summary>   Gets the Jacobian type. </summary>
        ///
        /// <value> The Jacobian type. </value>

        public JacobianType Type => m_Header->Type;

        /// <summary>   Gets or sets the Jacobian flags. </summary>
        ///
        /// <value> The Jacobian flags. </value>

        public JacobianFlags Flags
        {
            get => m_Header->Flags;
            set
            {
                // Some flags change the size of the jacobian; don't allow these to be changed:
                byte notPermitted = (byte)(JacobianFlags.EnableSurfaceVelocity | JacobianFlags.EnableMassFactors | JacobianFlags.EnableCollisionEvents);
                byte userFlags = (byte)value;
                byte alreadySet = (byte)m_Header->Flags;

                if ((notPermitted & (userFlags ^ alreadySet)) != 0)
                {
                    SafetyChecks.ThrowNotSupportedException("Cannot change flags which alter jacobian size");
                    return;
                }

                m_Header->Flags = value;
            }
        }

        /// <summary>   Gets a value indicating whether this object has mass factors. </summary>
        ///
        /// <value> True if this object has mass factors, false if not. </value>

        public bool HasMassFactors => m_Header->HasMassFactors;

        /// <summary>   Gets or sets the mass factors. </summary>
        ///
        /// <value> The mass factors. </value>

        public MassFactors MassFactors
        {
            get => m_Header->MassFactors;
            set
            {
                m_Header->MassFactors = value;
                ModifiersChanged = true;
            }
        }

        /// <summary>   Gets a value indicating whether this object has surface velocity. </summary>
        ///
        /// <value> True if this object has surface velocity, false if not. </value>

        public bool HasSurfaceVelocity => m_Header->HasSurfaceVelocity;

        /// <summary>   Gets or sets the surface velocity. </summary>
        ///
        /// <value> The surface velocity. </value>

        public SurfaceVelocity SurfaceVelocity
        {
            get => m_Header->SurfaceVelocity;
            set
            {
                m_Header->SurfaceVelocity = value;
                ModifiersChanged = true;
            }
        }

        /// <summary>   Gets angular jacobian. </summary>
        ///
        /// <param name="i">    Zero-based index of the jacobian. </param>
        ///
        /// <returns>   The angular jacobian. </returns>

        public ContactJacAngAndVelToReachCp GetAngularJacobian(int i)
        {
            return m_Header->AccessAngularJacobian(i);
        }

        /// <summary>   Sets angular jacobian. </summary>
        ///
        /// <param name="i">    Zero-based index of the jacobian. </param>
        /// <param name="j">    A ContactJacAngAndVelToReachCp to set. </param>

        public void SetAngularJacobian(int i, ContactJacAngAndVelToReachCp j)
        {
            m_Header->AccessAngularJacobian(i) = j;
            AngularChanged = true;
        }
    }

    /// <summary>   A modifiable contact jacobian. </summary>
    public unsafe struct ModifiableContactJacobian
    {
        internal ContactJacobian* m_ContactJacobian;

        /// <summary>   Gets a value indicating whether this object is modified. </summary>
        ///
        /// <value> True if modified, false if not. </value>
        public bool Modified { get; private set; }

        /// <summary>   Gets the number of contacts. </summary>
        ///
        /// <value> The total number of contacts. </value>
        public int NumContacts => m_ContactJacobian->BaseJacobian.NumContacts;

        /// <summary>   Gets or sets the normal. </summary>
        ///
        /// <value> The normal. </value>
        public float3 Normal
        {
            get => m_ContactJacobian->BaseJacobian.Normal;
            set
            {
                m_ContactJacobian->BaseJacobian.Normal = value;
                Modified = true;
            }
        }

        /// <summary>   Gets or sets the coefficient of friction. </summary>
        ///
        /// <value> The coefficient of friction. </value>
        public float CoefficientOfFriction
        {
            get => m_ContactJacobian->CoefficientOfFriction;
            set
            {
                m_ContactJacobian->CoefficientOfFriction = value;
                Modified = true;
            }
        }

        /// <summary>   Gets or sets the friction 0. </summary>
        ///
        /// <value> The friction 0. </value>
        public ContactJacobianAngular Friction0
        {
            get => m_ContactJacobian->Friction0;
            set
            {
                m_ContactJacobian->Friction0 = value;
                Modified = true;
            }
        }

        /// <summary>   Gets or sets the friction 1. </summary>
        ///
        /// <value> The friction 1. </value>
        public ContactJacobianAngular Friction1
        {
            get => m_ContactJacobian->Friction1;
            set
            {
                m_ContactJacobian->Friction1 = value;
                Modified = true;
            }
        }

        /// <summary>   Gets or sets the angular friction. </summary>
        ///
        /// <value> The angular friction. </value>
        public ContactJacobianAngular AngularFriction
        {
            get => m_ContactJacobian->AngularFriction;
            set
            {
                m_ContactJacobian->AngularFriction = value;
                Modified = true;
            }
        }
    }

    /// <summary>   A modifiable trigger jacobian. </summary>
    public struct ModifiableTriggerJacobian
    {
        internal unsafe TriggerJacobian* m_TriggerJacobian;
    }

    /// <summary>   The jacobians job extensions. </summary>
    public static class IJacobiansJobExtensions
    {
        /// <summary>   Schedules an IJacobiansJob for serial processing. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        /// <param name="job">      The scheduled job. </param>
        /// <param name="simulationSingleton">  The simulation singleton. </param>
        /// <param name="world">                [in,out] The physics world. </param>
        /// <param name="inputDeps">            The input dependencies. </param>
        ///
        /// <returns>   A JobHandle. </returns>
        public static JobHandle Schedule<T>(this T job, SimulationSingleton simulationSingleton, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IJacobiansJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }

            return ScheduleUnityPhysicsJacobiansJob(job, simulationSingleton.AsSimulation(), ref world, inputDeps);
        }


        /// <summary>   Schedules an IJacobiansJob for parallel processing. </summary>
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
            where T : struct, IJacobiansJobBase
        {
            // Should work only for UnityPhysics
            if (simulationSingleton.Type != SimulationType.UnityPhysics)
            {
                return inputDeps;
            }

            return ScheduleParallelUnityPhysicsJacobiansJob(job, innerLoopBatchCount, simulationSingleton.AsSimulation(), ref world, inputDeps);
        }

        static unsafe JobHandle ScheduleUnityPhysicsJacobiansJob<T>(T job, Simulation simulation, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IJacobiansJobBase
        {
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.PostCreateJacobians);

            if (simulation.StepContext.Jacobians.IsCreated)
            {
                var data = new JacobiansJobData<T>
                {
                    UserJobData = job,
                    JacobiansReader = simulation.StepContext.Jacobians.AsReader(),
                    Bodies = world.Bodies,
                    IsParallel = false
                };

                var jobReflectionData = JacobiansJobProcess<T>.jobReflectionData.Data;
                JacobiansJobProcess<T>.CheckReflectionDataCorrect(jobReflectionData);

                var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), jobReflectionData, inputDeps, ScheduleMode.Single);
                return JobsUtility.Schedule(ref parameters);
            }
            // else:

            return inputDeps;
        }

        static unsafe JobHandle ScheduleParallelUnityPhysicsJacobiansJob<T>(T job, int innerLoopBatchCount, Simulation simulation, ref PhysicsWorld world, JobHandle inputDeps)
            where T : struct, IJacobiansJobBase
        {
            SafetyChecks.CheckSimulationStageAndThrow(simulation.m_SimulationScheduleStage, SimulationScheduleStage.PostCreateJacobians);

            if (simulation.StepContext.Jacobians.IsCreated)
            {
                var jacobiansStream = simulation.StepContext.Jacobians;
                var data = new JacobiansJobData<T>
                {
                    UserJobData = job,
                    JacobiansReader = jacobiansStream.AsReader(),
                    Bodies = world.Bodies,
                    IsParallel = true
                };

                var jobReflectionData = JacobiansJobProcess<T>.jobReflectionData.Data;
                JacobiansJobProcess<T>.CheckReflectionDataCorrect(jobReflectionData);

                var parameters = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref data), jobReflectionData, inputDeps, ScheduleMode.Parallel);
                var forEachCountPtr = NativeStreamUnsafeUtility.GetUnsafeForEachCountPtr(ref jacobiansStream);
                var listDataPtr = (byte*)forEachCountPtr - sizeof(void*);
                return JobsUtility.ScheduleParallelForDeferArraySize(ref parameters, innerLoopBatchCount, listDataPtr, null);
            }
            // else:

            return inputDeps;
        }

        internal struct JacobiansJobData<T> where T : struct
        {
            public T UserJobData;
            public NativeStream.Reader JacobiansReader;

            // Disable aliasing restriction in case T has a NativeArray of PhysicsWorld.Bodies
            [ReadOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<RigidBody> Bodies;
            public bool IsParallel;
        }

        // Utility to help iterate over all the items in the jacobians job stream
        unsafe struct JacobiansJobIterator
        {
            NativeStream.Reader m_JacobiansReader;
            int m_CurrentForEachIndex;
            int m_ForEachIndexEnd;

            public JacobiansJobIterator(NativeStream.Reader reader, int forEachIndexBegin, int forEachIndexEnd)
            {
                SafetyChecks.CheckAreEqualAndThrow(true, forEachIndexBegin >= 0
                    && forEachIndexBegin <= forEachIndexEnd // Note: we use <= here since for empty readers,
                                                            // forEachIndexEnd will be identical to forEachIndexBegin,
                                                            // both being zero. This is still valid, and should not throw.
                    && forEachIndexEnd <= reader.ForEachCount);

                m_JacobiansReader = reader;
                m_CurrentForEachIndex = forEachIndexBegin;
                m_ForEachIndexEnd = forEachIndexEnd;

                MoveReaderToNextForEachIndex();
            }

            public bool HasItemsLeft => m_JacobiansReader.RemainingItemCount > 0;

            public JacobianHeader* ReadJacobianHeader()
            {
                int readSize = Read<int>();
                return (JacobianHeader*)Read(readSize);
            }

            byte* Read(int size)
            {
                byte* dataPtr = m_JacobiansReader.ReadUnsafePtr(size);
                MoveReaderToNextForEachIndex();
                return dataPtr;
            }

            ref T2 Read<T2>() where T2 : struct
            {
                int size = UnsafeUtility.SizeOf<T2>();
                return ref UnsafeUtility.AsRef<T2>(Read(size));
            }

            void MoveReaderToNextForEachIndex()
            {
                while (m_JacobiansReader.RemainingItemCount == 0 && m_CurrentForEachIndex < m_ForEachIndexEnd)
                {
                    m_JacobiansReader.BeginForEachIndex(m_CurrentForEachIndex++);
                }
            }
        }

        internal struct JacobiansJobProcess<T> where T : struct, IJacobiansJobBase
        {
            internal static readonly SharedStatic<IntPtr> jobReflectionData = SharedStatic<IntPtr>.GetOrCreate<JacobiansJobProcess<T>>();

            [Preserve]
            public static void Initialize()
            {
                if (jobReflectionData.Data == IntPtr.Zero)
                    jobReflectionData.Data = JobsUtility.CreateJobReflectionData(typeof(JacobiansJobData<T>), typeof(T), (ExecuteJobFunction)Execute);
            }

            [System.Diagnostics.Conditional("ENABLE_UNITY_COLLECTIONS_CHECK")]
            internal static void CheckReflectionDataCorrect(IntPtr reflectionData)
            {
                if (reflectionData == IntPtr.Zero)
                    SafetyChecks.ThrowInvalidOperationException("Reflection data was not set up by an Initialize() call");
            }

            public delegate void ExecuteJobFunction(ref JacobiansJobData<T> jobData, IntPtr additionalData,
                IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex);

            public unsafe static void Execute(ref JacobiansJobData<T> jobData, IntPtr additionalData,
                IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex)
            {
                while (true)
                {
                    int forEachIndexBegin = 0;
                    int forEachIndexEnd = jobData.JacobiansReader.ForEachCount;

                    if (jobData.IsParallel)
                    {
                        if (!JobsUtility.GetWorkStealingRange(ref ranges, jobIndex, out forEachIndexBegin,
                                out forEachIndexEnd))
                            break;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                        JobsUtility.PatchBufferMinMaxRanges(bufferRangePatchData, UnsafeUtility.AddressOf(ref jobData),
                            forEachIndexBegin, forEachIndexEnd - forEachIndexBegin);
#endif
                    }

                    var iterator = new JacobiansJobIterator(jobData.JacobiansReader, forEachIndexBegin, forEachIndexEnd);

                    while (iterator.HasItemsLeft)
                    {
                        JacobianHeader* header = iterator.ReadJacobianHeader();

                        var h = new ModifiableJacobianHeader
                        {
                            m_Header = header,
                            EntityPair = new EntityPair
                            {
                                EntityA = jobData.Bodies[header->BodyPair.BodyIndexA].Entity,
                                EntityB = jobData.Bodies[header->BodyPair.BodyIndexB].Entity
                            }
                        };
                        if (header->Type == JacobianType.Contact)
                        {
                            var contact = new ModifiableContactJacobian
                            {
                                m_ContactJacobian =
                                    (ContactJacobian*)UnsafeUtility.AddressOf(
                                        ref header->AccessBaseJacobian<ContactJacobian>())
                            };
                            jobData.UserJobData.Execute(ref h, ref contact);
                        }
                        else if (header->Type == JacobianType.Trigger)
                        {
                            var trigger = new ModifiableTriggerJacobian
                            {
                                m_TriggerJacobian =
                                    (TriggerJacobian*)UnsafeUtility.AddressOf(
                                        ref header->AccessBaseJacobian<TriggerJacobian>())
                            };

                            jobData.UserJobData.Execute(ref h, ref trigger);
                        }
                    }

                    // If we are not running in parallel, we are done.
                    if (!jobData.IsParallel)
                        break;
                }
            }
        }

        /// <summary>   Early job initialize. </summary>
        ///
        /// <typeparam name="T">    Generic type parameter. </typeparam>
        public static void EarlyJobInit<T>()
            where T : struct, IJacobiansJobBase
        {
            JacobiansJobProcess<T>.Initialize();
        }
    }
}
