# Hardware and Connections

This guide reflects the **checked-in code**, not a universal wiring diagram for every Arduino model.

| Device | Use | Connection |
| --- | --- | --- |
| Contact microphone | Captures throat-area acoustic activity for experiments | Audio-interface input 1 |
| M-Audio AIR 192/6 | Provides a host audio input | USB to macOS computer |
| MPU6050 | Acceleration and angular-velocity sensing | I²C to Arduino |
| Arduino UNO Q | Reads IMU and sends telemetry | USB to macOS computer |
| Computer | Python processing and dashboard | Both USB devices |

**Important:** The microphone is connected to the USB audio interface, **not** directly to Arduino. The current Python source searches for an M-Audio AIR 192/6 device name. A different interface requires changing the device-selection logic in both [event_detector_live.py](../event_detector_live.py) and [sensor_fusion_live.py](../sensor_fusion_live.py).

## Firmware

[firmware/dysphagrip_motion/dysphagrip_motion.ino](../firmware/dysphagrip_motion/dysphagrip_motion.ino) imports \`Arduino_RouterBridge.h\` and communicates through \`Monitor\`. The included sketch is intended for **Arduino UNO Q**, and is not drop-in compatible with an arbitrary Arduino board.

- I²C device address: \`0x68\`
- Sample interval target: \`40 ms\` (approximately 25 readings per second)
- Serial host expectation: \`115200\` baud
- Values emitted: acceleration X/Y/Z and gyro X/Y/Z
- Extra telemetry: sequence number and board sample timestamp

Example frame:

~~~text
Tilt X: -0.99 Y: 0.00 Z: 0.20 | Turn X: 1.0 Y: 2.0 Z: 3.0 seq=20 t_ms=8120
~~~

The code assumes the MPU6050 is available on the board's I²C bus. **Verify board-specific voltage, SDA/SCL pin mapping, grounding, and module ratings from the official board and module documentation before wiring.** Do not connect unsupported voltages to sensor pins.

## Physical safety

Do not constrict the neck or press electronics against sensitive tissue. The prototype is not suitable for unsupervised use by people who have swallowing difficulties. Do not ask a person with suspected dysphagia to swallow liquids or food as a DIY demonstration. Use informed consent and appropriate oversight for any research involving people.
