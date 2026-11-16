"""Mide la frecuencia de movimiento. Ejecuta: python test_arduino_connection.py"""

import re
from statistics import median
import time

from event_detector_live import IMU_PATTERN, choose_arduino_port, parse_imu_line


TELEMETRY = re.compile(r"(?:\|\s*)?seq=(\d{1,10})\s+t_ms=(\d{1,10})")
UINT32 = 1 << 32


def read_samples(port, warmup=4.0, duration=10.0):
    start = time.monotonic() + warmup
    end = start + duration
    samples = []
    rejected = 0
    while time.monotonic() < end:
        raw = port.read_until(b"\n", size=512)
        arrival = time.monotonic()
        if not raw or not start <= arrival < end:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not raw.endswith(b"\n") or parse_imu_line(line) is None:
            rejected += 1
            continue
        suffix = line[IMU_PATTERN.search(line).end():].strip()
        match = TELEMETRY.fullmatch(suffix)
        telemetry = tuple(map(int, match.groups())) if match else None
        if telemetry and max(telemetry) >= UINT32:
            telemetry = None
        samples.append((arrival, telemetry, line))
    return samples, rejected


def board_rate(samples):
    """Use consecutive valid packets; allow uint32 rollover, skip restarts."""
    steps = elapsed_ms = missing = 0
    for previous, current in zip(samples, samples[1:]):
        before, after = previous[1], current[1]
        if before is None or after is None:
            continue
        delta_seq = (after[0] - before[0]) % UINT32
        delta_ms = (after[1] - before[1]) % UINT32
        if 0 < delta_seq < UINT32 // 2 and 0 < delta_ms < UINT32 // 2:
            steps += delta_seq
            elapsed_ms += delta_ms
            missing += delta_seq - 1
    if elapsed_ms:
        return 1000 * steps / elapsed_ms, missing
    return None


def main():
    import serial
    from serial.tools import list_ports

    port_name = choose_arduino_port(list(list_ports.comports()))
    duration = 10.0
    print("Midiendo movimiento: 4 segundos de estabilización y 10 de medición.")
    try:
        with serial.Serial(port_name, 115200, timeout=0.2, exclusive=True) as port:
            samples, rejected = read_samples(port, duration=duration)
    except (OSError, serial.SerialException) as error:
        raise SystemExit(f"No se pudo leer {port_name}. Cierra el tablero y el monitor serie. {error}") from None

    print(f"Muestras recibidas: {len(samples)}. Líneas incompletas o sin lectura válida: {rejected}.")
    if len(samples) < 2:
        raise SystemExit("Se necesitan al menos dos muestras válidas para comparar los tiempos.")
    gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    print(f"Llegada a la Mac: {len(samples) / duration:.2f} Hz. Intervalo mediano: {median(gaps)*1000:.1f} ms.")
    print(f"Ejemplo: {samples[-1][2]}")
    result = board_rate(samples)
    if result:
        hz, missing = result
        print(f"Ritmo en la placa: {hz:.2f} Hz. Números de secuencia ausentes: {missing}.")
    else:
        print("Para medir el ritmo en la placa hacen falta dos lecturas con seq y t_ms válidos.")


if __name__ == "__main__":
    main()
