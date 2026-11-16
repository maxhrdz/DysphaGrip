# Setup, Running, and Experiments

This guide documents the Python desktop application on the repository's \`main\` branch.

## Prerequisites

- macOS and Python 3.13
- Arduino UNO Q running the [included firmware](../firmware/dysphagrip_motion/dysphagrip_motion.ino)
- MPU6050 connected via I²C
- USB audio interface recognized by the source code as **M-Audio AIR 192/6**
- Contact microphone on audio input 1
- USB cables and permission to access the microphone

## Installation

~~~bash
git clone https://github.com/maxhrdz/test2.git
cd test2
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

For each later terminal session:

~~~bash
source .venv/bin/activate
~~~

## Tests without connected hardware

~~~bash
python -m unittest discover -v
~~~

This runs the included unit-test suite with simulated test inputs. Actual test success depends on your local environment and is not asserted by this guide.

## Connect hardware

1. Flash the compatible Arduino firmware and connect the MPU6050 via I²C.
2. Connect the Arduino to the computer over USB.
3. Connect the contact microphone to input 1 of the M-Audio interface and the interface to the computer over USB.
4. Close the Arduino Serial Monitor, which may occupy the serial port.
5. Grant your terminal/Python runtime macOS microphone access when prompted.

## Start the dashboard

~~~bash
python sensor_fusion_live.py
~~~

If asked, select the Arduino serial port. Remain still and quiet for at least five seconds of initial calibration. Confirm microphone and MPU6050 are both connected; check that audio and motion traces respond to their respective inputs.

To monitor candidate events in the terminal:

~~~bash
python event_detector_live.py
~~~

Events may be appended to \`dysphagrip_events.csv\` in your current directory.

## Record a labeled experiment

~~~bash
python sensor_fusion_live.py --experiment
~~~

Click the dashboard window, then press the relevant digit **immediately after** an action:

| Key | Experimental label |
| --- | --- |
| 1 | Saliva swallow |
| 2 | Water swallow |
| 3 | Speaking |
| 4 | Head movement |
| 5 | Coughing |

Leave about five seconds between separate actions. Wait at least two seconds after the final mark before closing. These instructions are for appropriate, healthy-volunteer lab testing only; do not induce potentially unsafe swallowing.

The session under \`experiments/\` can include:

| File | Meaning |
| --- | --- |
| \`ground_truth.csv\` | Manually entered event labels (not clinician-verified ground truth) |
| \`predictions.csv\` | Detector candidate events |
| \`matches.csv\` | Temporal matches between labels and events |
| \`summary.csv\` | Experimental counts and errors |
| \`captures/mark_0001/audio.wav\` | Audio near a mark |
| \`captures/mark_0001/motion.csv\` | IMU readings near a mark |
| \`captures/mark_0001/metadata.json\` | Timing and capture metadata |

The default matching tolerance is 1.5 seconds. You may change it, for example:

~~~bash
python sensor_fusion_live.py --experiment --tolerance 1.0
~~~

## Arduino telemetry diagnostics

~~~bash
python test_arduino_connection.py
~~~

The sketch targets 25 Hz but the test reports *observed* delivery intervals and missing sequence IDs. The target must not be presented as an observed measurement.

## Troubleshooting

- **Microphone not found:** check USB, macOS permissions, and the hard-coded M-Audio device name in the source.
- **No Arduino port:** check the cable and drivers, select the correct device, and close Serial Monitor.
- **No IMU frames:** check I²C wiring and the expected \`Tilt X: ... | Turn X: ...\` serial format.
- **Detection remains in calibration:** confirm both streams are providing sufficient samples, and keep the sensors quiet.
- **False triggers:** verify placement, calibration conditions, and microphone gain; do not interpret a trigger as a diagnosis.

## Privacy

Audio captures and experiment labels can be sensitive. Do not commit participant recordings, metadata, or raw experiments into a public repository. Review \`.gitignore\` and obtain appropriate consent before collecting human-subject data.
