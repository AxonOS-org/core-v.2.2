# AxonOS Architecture

## Overview

AxonOS is a bare-metal real-time operating system for Cortex-M4F and Cortex-M33 targets, designed for closed-loop neurostimulation and motor-imagery BCI applications.

## Module Hierarchy

```
lib.rs
├── scheduler/      # EDF + Liu-Layland admission
├── signal/         # 6-stage zero-copy pipeline
├── ringbuf/        # Generic SPSC (Theorem 6.3)
├── ipc/            # DC1-DC6 dual-core contract
├── capability/     # Structural isolation (Theorem 8.3)
├── consent/        # FSM + stimulation interlock
├── attestation/    # ATECC608B HMAC
├── platform/       # DWT, GPIO
├── hal/            # Critical sections, barriers
└── zerocalib/      # Riemannian classifier
```

## Evidence Levels

| Level | Method | Hardware |
|-------|--------|----------|
| [L1]  | Instruction-count from assembly | None |
| [L2]  | DWT cycle counter | STM32F407 |
| [L3]  | Oscilloscope (Saleae Logic Pro 16) | STM32H573 |

## References

- Liu, C. L., & Layland, J. W. (1973). JACM 20(1), 46–61.
- Buttazzo, G. C. (2011). *Hard Real-Time Computing Systems* (3rd ed.). Springer.
- Vyukov, D. (2010). "Lock-free algorithms: The queue and the ring buffer."
- Miller, M. S., Yee, K., & Shapiro, J. (2003). "Capability myths demolished."
