#!/usr/bin/env python3
"""Extreme Value Theory (EVT) fitting for WCET estimation.

Fits Gumbel distribution to measured execution times to estimate
pWCET (probabilistic WCET) at 10^-9 probability level.
"""

import numpy as np
from scipy.stats import gumbel_r
import json
import argparse

def fit_gumbel(samples):
    """Fit Gumbel distribution and return pWCET at 10^-9."""
    loc, scale = gumbel_r.fit(samples)
    # pWCET at 10^-9: inverse CDF
    pwcet = gumbel_r.ppf(1 - 1e-9, loc=loc, scale=scale)
    return loc, scale, pwcet

def main():
    parser = argparse.ArgumentParser(description="EVT WCET fitting")
    parser.add_argument("input", help="JSON file with WCRT samples")
    parser.add_argument("--output", default="evt_results.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    samples = np.array(data["samples"])
    loc, scale, pwcet = fit_gumbel(samples)

    print(f"Gumbel loc: {loc:.2f} µs")
    print(f"Gumbel scale: {scale:.2f} µs")
    print(f"pWCET (10^-9): {pwcet:.2f} µs")

    with open(args.output, "w") as f:
        json.dump({
            "loc": loc,
            "scale": scale,
            "pwcet_1e9": pwcet,
            "samples": len(samples),
        }, f, indent=2)

if __name__ == "__main__":
    main()
