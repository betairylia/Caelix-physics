using UnityEngine;
using UnityEngine.InputSystem;

public class CameraControl : MonoBehaviour
{
    public float lookSpeedH = 2f;
    public float lookSpeedV = 2f;
    public float zoomSpeed = 2f;
    public float dragSpeed = 5f;

    private float yaw;
    private float pitch;


    private void Start()
    {
        // x - right    pitch
        // y - up       yaw
        // z - forward  roll
        yaw = transform.eulerAngles.y;
        pitch = transform.eulerAngles.x;
    }

    void Update()
    {
        if (!enabled) return;

        // Touchscreen controls:
        if (Touchscreen.current != null && Touchscreen.current.touches.Count > 0)
        {
            const float kTouchToMouseScale = 0.25f;
            // look around with first touch
            var t0 = Touchscreen.current.touches[0];
            if (t0.isInProgress)
            {
                yaw += lookSpeedH * kTouchToMouseScale * t0.delta.ReadValue().x;
                pitch -= lookSpeedV * kTouchToMouseScale * t0.delta.ReadValue().y;
                transform.eulerAngles = new Vector3(pitch, yaw, 0f);
            }

            // Second touch → move camera
            if (Touchscreen.current.touches.Count > 1)
            {
                var t1 = Touchscreen.current.touches[1];
                if (t1.isInProgress)
                {
                    Vector2 delta = t1.delta.ReadValue();
                    Vector3 offset = new Vector3(delta.x, 0, delta.y);
                    transform.Translate(offset * (Time.deltaTime * kTouchToMouseScale), Space.Self);
                }
            }
        }

        // Mouse controls:
        if (Mouse.current != null)
        {
            // Look around with Right Mouse
            if (Mouse.current.rightButton.isPressed)
            {
                Vector2 mouseDelta = Mouse.current.delta.ReadValue();
                yaw += lookSpeedH * mouseDelta.x;
                pitch -= lookSpeedV * mouseDelta.y;

                transform.eulerAngles = new Vector3(pitch, yaw, 0f);

                Vector3 offset = Vector3.zero;
                float offsetDelta = Time.deltaTime * dragSpeed;
                if (Keyboard.current.leftShiftKey.isPressed) offsetDelta *= 5.0f;
                if (Keyboard.current.sKey.isPressed) offset.z -= offsetDelta;
                if (Keyboard.current.wKey.isPressed) offset.z += offsetDelta;
                if (Keyboard.current.aKey.isPressed) offset.x -= offsetDelta;
                if (Keyboard.current.dKey.isPressed) offset.x += offsetDelta;
                if (Keyboard.current.qKey.isPressed) offset.y -= offsetDelta;
                if (Keyboard.current.eKey.isPressed) offset.y += offsetDelta;

                transform.Translate(offset, Space.Self);
            }

            // Drag camera around with Middle Mouse
            if (Mouse.current.middleButton.isPressed)
            {
                Vector2 drag = Mouse.current.delta.ReadValue();
                transform.Translate(-drag.x * Time.deltaTime * dragSpeed, -drag.y * Time.deltaTime * dragSpeed, 0);
            }

            // Zoom in and out with Mouse Wheel
            float scroll = Mouse.current.scroll.ReadValue().y;
            transform.Translate(0, 0, scroll * zoomSpeed * Time.deltaTime, Space.Self);
        }
    }
}
