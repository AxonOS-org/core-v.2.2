# Zero-Copy Signal Path

## Stages

1. Kalman state estimator (8-ch)
2. FIR bandpass (order 64, 8-ch)
3. Notch filter (50 Hz + 60 Hz)
4. Artifact rejection (±120 µV)
5. CSP spatial filter (8×8)
6. LDA classifier

## WCET

| Stage | C_i (µs) |
|-------|----------|
| Kalman | 80.0 |
| FIR | 320.0 |
| Notch | 60.0 |
| Artifact | 40.0 |
| CSP | 100.0 |
| LDA | 40.2 |
| **Total** | **640.2** |

## Zero-Copy Guarantee

SPSC ring buffer passes `MotorImageryClass` by value (Copy type). No heap allocation on hot path.

## References

- Blankertz, B., et al. (2008). "Optimizing spatial filters for robust EEG single-trial analysis." IEEE SPM 25(1), 41–56.
- Fukunaga, K. (1990). *Introduction to Statistical Pattern Recognition* (2nd ed.). Academic Press.
- Welch, G., & Bishop, G. (2006). "An Introduction to the Kalman Filter." UNC TR 95-041.
