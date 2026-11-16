# Safety, Ethics, and Limitations

**DysphaGrip is an experimental engineering prototype and is not a medical device.**

## What it does

The checked-in implementation captures microphone and MPU6050 signals and identifies **candidate swallowing-related events** when both sensor features satisfy calibrated thresholds within a short time window. It also supports manual event labeling and experimental comparison.

## What it does not do

- It does not diagnose dysphagia.
- It does not detect, prevent, or rule out aspiration, including **silent aspiration**.
- It does not distinguish a safe swallow from an unsafe swallow.
- It does not replace an evaluation by a speech-language pathologist, physician, or other qualified clinician.
- It does not provide clinically established accuracy, sensitivity, specificity, or reliability.
- It does not make decisions about diet, medication, swallowing therapy, or care.

## Known limitations

- Talking, coughing, head movement, sensor slippage, or environmental vibration may produce correlated signals.
- Some real swallows may not exceed the detector's thresholds.
- Thresholds calibrated in one environment or on one person may not transfer to others.
- Data come from different USB acquisition paths, which can introduce timing jitter.
- Manual labels are not equivalent to validated clinical ground truth.
- No evidence of clinical validation, a representative dataset, or regulatory clearance is provided in this repository.

## Human participant safety

Do not test oral intake on people with suspected dysphagia without clinical supervision. Do not tighten or obstruct the neck with a sensor or collar. Treat personal health information and audio recordings as sensitive: obtain consent, avoid public uploads, and follow applicable privacy and research review requirements.

## Responsible communication

Use the language **candidate event**, **experimental sensor fusion**, and **research prototype**. Avoid phrases such as “detects silent aspiration,” “monitors swallowing safety,” or “clinically accurate AI diagnosis.”

If someone has difficulty swallowing, choking, or breathing, seek appropriate medical care rather than relying on this prototype.
