# System Architecture

DysphaGrip is a local, two-stream sensor-fusion prototype. The desktop application is written in Python.

## Data flow

~~~mermaid
flowchart TD
    A["Contact microphone"] --> B["M-Audio AIR 192/6 USB interface"]
    B --> C["Computer audio input (sounddevice)"]
    D["MPU6050: accel + gyro"] --> E["Arduino UNO Q (I²C)"]
    E --> F["USB serial (pyserial, 115200 baud)"]
    C --> G["SwallowEventDetector"]
    F --> G
    G --> H["Candidate event log (CSV)"]
    G --> I["Matplotlib live dashboard"]
    C --> J["Optional experiment capture"]
    F --> J
    J --> K["Manual labels and validation report"]
~~~

The audio interface and Arduino are two *separate* USB inputs to the computer.

## Detection logic

1. Read audio blocks and compute root-mean-square (RMS) amplitude and peak level.
2. Parse the serial frame for 3-axis acceleration and angular velocity.
3. Compute changes in acceleration and gyroscope magnitude.
4. During a quiet calibration period (at least 5 seconds), estimate noise-dependent thresholds.
5. Register a candidate only when qualifying audio and IMU triggers occur within a **0.65 s** coincidence window.
6. Apply a default **1.5 s** cooldown to reduce repeated counts.
7. Append candidate events to a CSV file and update the local dashboard.

The detector implementation is in [event_detector_live.py](../event_detector_live.py). These are heuristic thresholds, **not a diagnosis or clinical aspiration detector**.

## Experimental validation

When launched with the \`--experiment\` flag, the desktop application permits manual labels for five actions. [label_capture.py](../label_capture.py) collects audio and IMU samples around marked moments (two seconds before and after). [validation.py](../validation.py) performs one-to-one temporal matching with a default **1.5-second** tolerance and writes reports. Labels are manually entered; their presence does not establish medical ground truth.

## Code map

- [event_detector_live.py](../event_detector_live.py): detector class, serial parser, and audio input.
- [sensor_fusion_live.py](../sensor_fusion_live.py): Matplotlib dashboard, experiment UI, and sampling coordination.
- [label_capture.py](../label_capture.py): labeled audio and motion capture.
- [validation.py](../validation.py): comparison between manual labels and candidate events.
- [firmware](../firmware/dysphagrip_motion/dysphagrip_motion.ino): MPU6050 acquisition and serial telemetry.

## Constraints and trade-offs

- Dual USB acquisition is easy to prototype but has host-side timing jitter.
- A microphone plus IMU can reject some single-sensor activity, but coincident unrelated signals can still produce false positives.
- Calibration depends on local noise and motion; changing sensor placement or environment may invalidate thresholds.
- The graphical interface and audio interface make the checked-in implementation desktop-based, not an embedded standalone wearable.
- A static project website is separate from the actual local monitoring application.
