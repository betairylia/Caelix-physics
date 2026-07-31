using UnityEngine;

namespace Unity.Physics.Authoring
{
    /// <summary>
    /// Authoring component that defines which type of solver to use for physics calculations,
    /// including contacts and joints, allowing the physics engine to prioritize either
    /// accuracy or performance for specific bodies.
    /// </summary>
    [Icon(k_IconPath)]
    [AddComponentMenu("Entities/Physics/Solver Type")]
    [HelpURL(HelpURLs.SolverTypeAuthoring)]
    [DisallowMultipleComponent]
    public class SolverTypeAuthoring : MonoBehaviour
    {
        const string k_IconPath = "Packages/com.unity.physics/Unity.Physics.Editor/Editor Default Resources/Icons/d_Solver@64.png";

        /// <summary>
        /// <para> Specifies the type of solver used for joints attached to this body.</para>
        /// <para> Use <b>Direct</b> for higher accuracy and <b>Iterative</b> for better performance.
        /// If no Solver Type component is attached, <b>Iterative</b> is used by default.</para>
        /// </summary>
        [Tooltip("Specifies the type of solver used for joints attached to this body.\n" +
            "Use <b>Direct</b> for higher accuracy and <b>Iterative</b> for better performance. " +
            "If no Solver Type component is attached, <b>Iterative</b> is used by default.")]
        public SolverType JointSolverType = Solver.kDefaultSolverType;

        /// <summary>
        /// <para> Specifies the type of solver used for contacts with this body.</para>
        /// <para> Use <b>Direct</b> for higher accuracy and <b>Iterative</b> for better performance.
        /// If no Solver Type component is attached, <b>Iterative</b> is used by default.</para>
        /// </summary>
        [Tooltip("Specifies the type of solver used for contacts with this body.\n" +
            "Use <b>Direct</b> for higher accuracy and <b>Iterative</b> for better performance. " +
            "If no Solver Type component is attached, <b>Iterative</b> is used by default.")]
        public SolverType ContactSolverType = Solver.kDefaultSolverType;

        private void Start()
        {
        }
    }
}
