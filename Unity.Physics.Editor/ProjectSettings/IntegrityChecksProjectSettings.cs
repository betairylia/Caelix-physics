using System;
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEditor.Build;
using UnityEditorInternal;
using UnityEngine.UIElements;

namespace Unity.Physics.Editor.ProjectSettings
{
    /// <summary>
    /// Provides editor-only helpers for Unity Physics project settings by managing
    /// scripting define symbols across relevant build target groups.
    /// Exposes two toggles via static properties: IntegrityChecksDisabled and DebugDisplayRuntimeEnabled.
    /// </summary>
    public static class Preferences
    {
        const string k_DisableIntegrityDefine = "UNITY_PHYSICS_DISABLE_INTEGRITY_CHECKS";
        const string k_EnableDebugDisplayAtRuntime = "ENABLE_UNITY_PHYSICS_RUNTIME_DEBUG_DISPLAY";

        const char k_defineSeparator = ';';

        /// <summary>
        /// Controls whether integrity checks are disabled.
        /// Disable integrity checks when measuring performance.
        /// Enable them when validating simulation quality and behaviour.
        /// </summary>
        public static bool IntegrityChecksDisabled
        {
            get => DefineExists(k_DisableIntegrityDefine);
            set => UpdateDefine(k_DisableIntegrityDefine, value);
        }

        /// <summary>
        /// DebugDisplayRuntimeEnabled
        /// </summary>
        public static bool DebugDisplayRuntimeEnabled
        {
            get => DefineExists(k_EnableDebugDisplayAtRuntime, BuildPipeline.GetBuildTargetGroup(EditorUserBuildSettings.activeBuildTarget));
            set => UpdateDefine(k_EnableDebugDisplayAtRuntime, value);
        }

        private static void UpdateDefine(string define, bool add)
        {
            //collect all relevant build targets
            var buildTargetGroups = new List<BuildTargetGroup>();

            var activeBuildTarget = EditorUserBuildSettings.activeBuildTarget;
            var activeBuildTargetGroup = BuildPipeline.GetBuildTargetGroup(activeBuildTarget);

            // Provide the define for activeBuildTargetGroup(e.g. Android, PS4)
            buildTargetGroups.Add(activeBuildTargetGroup);

            // Windows, Mac, Linux - always include these, as they are the only ones where the development happens
            // and could possibly want/not want integrity checks in the editor, as opposed to only the connected device.
            if (activeBuildTargetGroup != BuildTargetGroup.Standalone)
            {
                buildTargetGroups.Add(BuildTargetGroup.Standalone);
            }

            foreach (var buildTargetGroup in buildTargetGroups)
            {
                var fromBuildTargetGroup = NamedBuildTarget.FromBuildTargetGroup(buildTargetGroup);
                var defines = PlayerSettings.GetScriptingDefineSymbols(fromBuildTargetGroup);

                // We add the separator at the end so we can add a new one if needed
                // Unity will automatically remove any unneeded separators at the end
                if (defines.Length > 0 && !defines.EndsWith("" + k_defineSeparator))
                    defines = defines + k_defineSeparator;

                var definesSb = new StringBuilder(defines);

                if (add)
                {
                    // add at the end if it isn't already defined
                    if (!defines.Contains(define))
                    {
                        definesSb.Append(define);
                        definesSb.Append(k_defineSeparator);
                    }
                }
                else
                {
                    // find it and just replace that spot with and empty string
                    var replaceToken = define + k_defineSeparator;
                    definesSb.Replace(replaceToken, "");
                }

                PlayerSettings.SetScriptingDefineSymbols(fromBuildTargetGroup, definesSb.ToString());
            }
        }

        private static bool DefineExists(string define)
        {
            var fromBuildTargetGroup = NamedBuildTarget.FromBuildTargetGroup(BuildTargetGroup.Standalone);
            var defines = PlayerSettings.GetScriptingDefineSymbols(fromBuildTargetGroup);
            return defines.Contains(define);
        }

        private static bool DefineExists(string define, BuildTargetGroup buildTargetGroup)
        {
            var fromBuildTargetGroup = NamedBuildTarget.FromBuildTargetGroup(buildTargetGroup);
            var defines = PlayerSettings.GetScriptingDefineSymbols(fromBuildTargetGroup);
            return defines.Contains(define);
        }
    }

    /// <summary>
    /// Bridges Unity Physics project settings into the built-in Physics Manager settings inspector
    /// </summary>
    public class PhysicsManagerInspectorInjector
    {
        /// <summary>
        /// Static InitializeOnLoadMethod that registers a ProjectSettingsEcsExtension with PhysicsManagerInspectorBridge,
        /// enabling custom UI elements (toggles for integrity checks and runtime debug display) to appear in the Physics Manager’s settings tab.
        /// </summary>
        [InitializeOnLoadMethod]
        public static void Register()
        {
            PhysicsManagerInspectorBridge.RegisterECSInspectorExtension(new ProjectSettingsEcsExtension());
        }
    }

    class ProjectSettingsEcsExtension : IPhysicsProjectSettingsECSInspectorExtension
    {
        public void SetupMainPageItems(DropdownField dropDown, HelpBox infoBox, HelpBox warningBox, SerializedObject physicsManager)
        {
            dropDown.visible = false;
            infoBox.visible = false;
            warningBox.visible = false;
        }

        VisualElement CreateAlignedToggle(string labelText, string tooltip, bool initialValue, Action<bool> onChanged)
        {
            // Row container
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.alignItems = Align.Center;

            // Label with Unity toggle styling
            var label = new Label(labelText);
            label.tooltip = tooltip;
            label.style.flexShrink = 0;
            label.style.width = 210; // Fixed width for alignment

            // Toggle (without text)
            var toggle = new Toggle();
            toggle.tooltip = tooltip;
            toggle.SetValueWithoutNotify(initialValue);
            toggle.RegisterValueChangedCallback(evt => onChanged(evt.newValue));

            // Add label first, then toggle
            row.Add(label);
            row.Add(toggle);

            return row;
        }

        public void SetupSettingsTab(Tab ecsTab, SerializedObject physicsManager)
        {
            var ecsContent = ecsTab.Q(name: "tab-content__ecs");

            ecsContent.Add(CreateAlignedToggle(
                "Enable Integrity Checks",
                "Integrity checks should be disabled when measuring performance. Integrity checks should be enabled when checking simulation quality and behaviour.",
                !Preferences.IntegrityChecksDisabled,
                value => Preferences.IntegrityChecksDisabled = !value
            ));

            ecsContent.Add(CreateAlignedToggle(
                "Enable Player Debug Display",
                "Allows debugging physics directly in the Player build. Enable to inspect behavior in-game. Disable for better performance.",
                Preferences.DebugDisplayRuntimeEnabled,
                value => Preferences.DebugDisplayRuntimeEnabled = value
            ));
        }
    }
}
