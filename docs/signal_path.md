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
