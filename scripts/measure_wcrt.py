#!/usr/bin/env python3
"""Measure WCRT via DWT cycle counter and fit EVT distribution.

Usage:
    python measure_wcrt.py --port /dev/ttyACM0 --epochs 1000000

References:
    - Castillo, E. (1988). *Extreme Value Theory in Engineering*.
      Academic Press. [EVT fitting]
    - AxonOS RFC-0003: Evidence Taxonomy.
"""

import argparse
import serial
import numpy as np
from scipy.stats import genextreme

def main():
    parser = argparse.ArgumentParser(description="Measure AxonOS WCRT")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    samples = []

    print(f"Collecting {args.epochs} epochs...")
    for _ in range(args.epochs):
        line = ser.readline().decode().strip()
        if line.startswith("WCET:"):
            samples.append(int(line.split(":")[1]))

    samples = np.array(samples)
    print(f"Mean: {samples.mean():.1f} µs")
    print(f"Max:  {samples.max()} µs")

    shape, loc, scale = genextreme.fit(samples)
    p99 = genextreme.ppf(0.99, shape, loc=loc, scale=scale)
    print(f"EVT p99: {p99:.1f} µs")

if __name__ == "__main__":
    main()
