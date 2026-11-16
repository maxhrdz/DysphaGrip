# DysphaGrip

**Exploring swallowing-related event detection through acoustic sensing, motion tracking, and real-time sensor fusion.**

[Project website](docs/index.html) · [System architecture](docs/ARCHITECTURE.md) · [Hardware](docs/HARDWARE.md) · [Setup](docs/SETUP.md) · [Safety](docs/LIMITATIONS.md)

**SanD Hacks 2026 · UC San Diego · Experimental research prototype**

> **Not a medical device.** DysphaGrip is an engineering proof of concept. A “possible swallow event” is a sensor-triggered candidate, not evidence of safe swallowing, dysphagia, or aspiration. Never use it for clinical decisions.

## The problem

Swallowing difficulties (dysphagia) can affect people recovering from a stroke or living with neurological conditions. Assessing swallowing requires qualified clinicians and, when indicated, clinical instruments. DysphaGrip explores whether inexpensive, complementary sensors can capture patterns associated with swallowing events for future research.

## What we built

DysphaGrip combines two independent data streams:

- **Acoustic signal:** a contact microphone connected through a USB audio interface to a computer.
- **Motion signal:** three-axis acceleration and three-axis angular velocity from an MPU6050, streamed by an Arduino over USB serial.

Python processes both streams, calibrates against a short quiet baseline, and records a candidate event when an audio trigger and a motion trigger fall within a **0.65-second coincidence window** (default). A cooldown helps reduce repeated counts. The desktop dashboard displays waveforms, motion traces, connection status, and an event history.

This is **rule-based sensor fusion**, not a clinically validated machine-learning classifier.

## Features

| Feature | Current implementation |
| --- | --- |
| Live sensor monitoring | Audio via sounddevice; serial motion via pyserial |
| Adaptive calibration | Baseline-derived thresholds with a minimum five-second calibration |
| Candidate-event detection | Audio–IMU coincidence and cooldown |
| Desktop visualization | Matplotlib plots and event history |
| Experimental labeling | Saliva, water, speaking, head movement, and coughing |
| Experimental evaluation | Label-to-event matching and session reports; not clinical validation |
| Automated tests | Python unittest suite |
| Hardware diagnostics | Arduino sample-rate and serial-frame inspection |

## Architecture

~~~mermaid
flowchart LR
    MIC["Contact microphone"] --> USB["USB audio interface"]
    IMU["MPU6050 · 6 axes"] --> MCU["Arduino · I²C + serial"]
    USB --> HOST["Python acquisition on computer"]
    MCU --> HOST
    HOST --> FEATURES["Audio RMS + IMU features"]
    FEATURES --> FUSION["Adaptive calibration + time-window fusion"]
    FUSION --> DASH["Desktop dashboard"]
    FUSION --> CSV["Candidate-event CSV"]
    HOST --> CAPTURE["Optional labeled captures"]
~~~

**The microphone is not connected to the Arduino.** Both devices connect to the computer through separate input paths. See [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Hardware

| Component | Responsibility |
| --- | --- |
| Contact microphone | Captures acoustic/vibration activity |
| M-Audio AIR 192/6 USB audio interface | Provides the audio input explicitly selected in the current Python code |
| MPU6050 accelerometer + gyroscope | Measures 3 acceleration and 3 rotation axes |
| Arduino UNO Q | Streams IMU data using the included firmware |
| Computer running Python | Processes signals, logs events, and renders the UI |

The checked-in sketch imports Arduino_RouterBridge.h and targets **Arduino UNO Q**. Other boards may require firmware changes. See [HARDWARE.md](docs/HARDWARE.md).

## Quick start (macOS)

Requirements: Python 3.13, the connected Arduino and MPU6050, a contact microphone with compatible audio interface, and macOS device permissions.

~~~bash
git clone https://github.com/maxhrdz/test2.git
cd test2
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

1. Upload the [included Arduino firmware](firmware/dysphagrip_motion/dysphagrip_motion.ino) to the supported board and connect the MPU6050 through I²C.
2. Connect the Arduino and audio interface by USB, with the microphone plugged into input 1.
3. Close the Arduino Serial Monitor before launching the Python application.
4. Remain still and quiet during calibration, then confirm both inputs report **Connected**.

Open the desktop dashboard:

~~~bash
python sensor_fusion_live.py
~~~

Or use the command-line event detector:

~~~bash
python event_detector_live.py
~~~

Run an experiment with manual labels:

~~~bash
python sensor_fusion_live.py --experiment
~~~

In experiment mode, click the dashboard and press **1** for saliva swallow, **2** for water swallow, **3** for speaking, **4** for head movement, or **5** for coughing. Each mark records approximately two seconds of data before and after it. Allow time after the final mark for the capture to finish.

For troubleshooting and a breakdown of generated files, see [SETUP.md](docs/SETUP.md).

## Tests

The unit tests do not require attached hardware:

~~~bash
python -m unittest discover -v
~~~

Diagnose the Arduino's sample delivery rate:

~~~bash
python test_arduino_connection.py
~~~

The included firmware requests a sample approximately every **40 ms (25 Hz)**. This is a target, not a reported experimental result.

## Repository map

~~~text
.
├── README.md
├── requirements.txt
├── event_detector_live.py           # Detector and input integration
├── sensor_fusion_live.py            # Desktop dashboard and experiments
├── label_capture.py                 # Time-aligned audio / motion captures
├── validation.py                    # Experimental label matching
├── test_*.py                        # Tests and diagnostic scripts
├── firmware/
│   └── dysphagrip_motion/
│       └── dysphagrip_motion.ino
└── docs/
    ├── index.html                   # Static project showcase
    ├── ARCHITECTURE.md
    ├── HARDWARE.md
    ├── SETUP.md
    └── LIMITATIONS.md
~~~

## Evaluation and limitations

Labeled sessions can generate ground_truth.csv, predictions.csv, matches.csv, summary.csv, and per-label captures under experiments/. These artifacts help examine false triggers, missed labeled events, and event timing. **No validated sensitivity, specificity, accuracy, or clinical performance is claimed here.**

Talking, coughing, head movement, sensor shifts, and environmental noise may cause false triggers. A missed event does not mean swallowing was safe. DysphaGrip **does not detect or rule out silent aspiration**. Read [LIMITATIONS.md](docs/LIMITATIONS.md).

## Roadmap

- Collect appropriately consented and carefully labeled data across people and environments.
- Measure false positives, missed candidate events, and signal quality with a repeatable protocol.
- Improve synchronization and sensor-placement consistency.
- Compare rule-based fusion to machine-learning methods after adequate data collection.
- Explore a smaller wearable form factor with appropriate clinical and engineering collaboration.

These are proposed next steps, not completed features.

## Website

The static showcase lives in [docs/index.html](docs/index.html). To publish it, open **GitHub Settings → Pages → Deploy from a branch → main → /docs → Save**. GitHub Pages will display a public URL after deployment. The showcase is informational; the live sensor dashboard runs locally in Python.

## Credits

Created as a team project at **SanD Hacks 2026**, hosted at **UC San Diego**. Add individual team credits after confirming the appropriate attributions.

## License

No license has been selected. Public repository visibility does not itself grant permission to reuse the code; add a license after the contributors agree.
