#if UNITY_EDITOR
using System;
using UnityEditor;
using UnityEngine;

namespace Unity.Physics.Authoring
{
    [CustomEditor(typeof(PhysicsDebugDisplayAuthoring))]
    public class PhysicsDebugDisplayAuthoringEditor : Editor
    {
        private GUIStyle enabledStyle;
        private GUIContent visibleIcon;
        private GUIContent hiddenIcon;
        private bool initialized = false;

        void OnEnable()
        {
            initialized = false;
        }

        public override void OnInspectorGUI()
        {
            if (!initialized)
            {
                // Note: lazy initialization here rather than directly in OnEnable to avoid nullptr access in EditorStyles after domain reload
                enabledStyle = new GUIStyle(EditorStyles.label)
                {
                    normal = { textColor = Color.green }
                };

                visibleIcon = EditorGUIUtility.IconContent("VisibilityOn");
                hiddenIcon = EditorGUIUtility.IconContent("VisibilityOff");

                initialized = true;
            }

            serializedObject.Update();

            var drawFlags = new(string property, string label)[]
            {
                ("DrawColliders", "Draw Colliders"),
                ("DrawColliderEdges", "Draw Collider Edges"),
                ("DrawColliderAabbs", "Draw Collider AABBs"),
                ("DrawMassProperties", "Draw Mass Properties"),
                ("DrawBroadphase", "Draw Broadphase"),
                ("DrawContacts", "Draw Contacts"),
                ("DrawCollisionEvents", "Draw Collision Events"),
                ("DrawTriggerEvents", "Draw Trigger Events"),
                ("DrawJoints", "Draw Joints"),
            };

            DrawSeparator("Debug Display Options");

            foreach (var(property, label) in drawFlags)
            {
                var prop = serializedObject.FindProperty(property);
                DrawToggleWithEye(prop, label);
            }

            DrawSeparator("Constraint Graph");

            drawFlags = new(string property, string label)[]
            {
                ("DisplayDirectSolverIslands", "Draw Direct Solver Islands"),
#if UNITY_PHYSICS_DISPLAY_ADVANCED_SOLVER_DATA
                ("DisplayDirectSolverIslandsIndex", "Draw Direct Solver Islands Index"),
#endif
                ("DrawIterativeSolverPhases", "Draw Iterative Solver Phases"),
#if UNITY_PHYSICS_DISPLAY_ADVANCED_SOLVER_DATA
                ("DrawIterativeSolverPhaseIndex", "Draw Iterative Solver Phase Index"),
#endif
            };

            var drawDirectSolverIslandsProp = serializedObject.FindProperty("DisplayDirectSolverIslands");
            var drawIterativeSolverPhasesProp = serializedObject.FindProperty("DrawIterativeSolverPhases");

            foreach (var(property, label) in drawFlags)
            {
                var prop = serializedObject.FindProperty(property);
                bool enabled = true;

                if (property == "DisplayDirectSolverIslandsIndex")
                    enabled = drawDirectSolverIslandsProp.boolValue;

                if (property == "DrawIterativeSolverPhaseIndex")
                    enabled = drawIterativeSolverPhasesProp.boolValue;

                using (new EditorGUI.DisabledScope(!enabled))
                {
                    DrawToggleWithEye(prop, label);
                }
            }


            DrawSeparator("Integration Mode");

            EditorGUILayout.PropertyField(serializedObject.FindProperty("ColliderDisplayMode"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("ColliderEdgesDisplayMode"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("ColliderAabbDisplayMode"));

            serializedObject.ApplyModifiedProperties();
        }

        void DrawToggleWithEye(SerializedProperty prop, string label)
        {
            EditorGUILayout.BeginHorizontal();

            var tooltip = GetTooltip(prop.name);
            var icon = prop.boolValue ? visibleIcon : hiddenIcon;
            var color = GUI.color;
            GUI.color = prop.boolValue ? Color.green : Color.gray;

            var iconContent = new GUIContent(icon.image, tooltip);
            if (GUILayout.Button(iconContent, GUILayout.Width(20), GUILayout.Height(18)))
            {
                prop.boolValue = !prop.boolValue;
            }

            GUI.color = color;

            var labelContent = new GUIContent(label, tooltip);
            GUILayout.Label(labelContent, prop.boolValue ? enabledStyle : EditorStyles.label);

            EditorGUILayout.EndHorizontal();
        }

        void DrawSeparator(string title)
        {
            EditorGUILayout.Space(10);

            // Line above
            var rect = EditorGUILayout.GetControlRect(false, 2);
            rect.height = 1;
            EditorGUI.DrawRect(rect, new Color(0.3f, 0.3f, 0.3f, 1));

            // Title centered
            EditorGUILayout.LabelField(title, EditorStyles.boldLabel);

            // Line below (optional)
            rect = EditorGUILayout.GetControlRect(false, 2);
            rect.height = 1;
            EditorGUI.DrawRect(rect, new Color(0.3f, 0.3f, 0.3f, 1));

            EditorGUILayout.Space(5);
        }

        string GetTooltip(string propertyName)
        {
            var targetType = typeof(PhysicsDebugDisplayAuthoring);
            var field = targetType.GetField(propertyName);
            if (field != null && Attribute.GetCustomAttribute(field, typeof(TooltipAttribute)) is TooltipAttribute tooltip)
                return tooltip.tooltip;
            return string.Empty;
        }
    }
}
#endif
