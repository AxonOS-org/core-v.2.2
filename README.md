# AxonOS Kernel v0.2.4

**Safety-critical `#![no_std]` Rust microkernel for brain-computer interface systems**

AxonOS is a bare-metal real-time operating system for Cortex-M4F and Cortex-M33 targets, designed for closed-loop neurostimulation and motor-imagery BCI applications.

## Key Features

- **EDF Scheduling**: Earliest-Deadline-First with Liu-Layland schedulability proof
- **Zero-Copy Signal Path**: Generic SPSC ring buffer from ADC DMA to classifier
- **Capability Isolation**: Structural data minimisation at type-system level
- **Dual-Core Contract**: Formal timing contract between M4F DSP and A53 app core
- **Targeted Unsafe**: `#![deny(unsafe_code)]` crate-wide; explicit `unsafe`
  only in `ringbuf::spsc` with proof invariants

## Evidence Levels

| Level | Method | Hardware |
|-------|--------|----------|
| [L1]  | Instruction-count from assembly | None |
| [L2]  | DWT cycle counter | STM32F407 |
| [L3]  | Oscilloscope (Saleae Logic Pro 16) | STM32H573 |

## Quick Start

```bash
git clone https://github.com/AxonOS-org/axonos-kernel.git
cd axonos-kernel

# Build for STM32F407 (Cortex-M4F)
cargo build --target thumbv7em-none-eabihf --features cortex-m4f

# Build for STM32H573 (Cortex-M33 with TrustZone)
cargo build --target thumbv8m.main-none-eabihf --features cortex-m33,trustzone

# Run tests on host
cargo test --lib

# Run Kani proofs
cargo kani --features kani
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Application Layer                     │
│  (Cortex-A53: Session, BLE/Wi-Fi, WASM sandbox)           │
├─────────────────────────────────────────────────────────────┤
│                      IPC Contract (DC1-DC6)               │
│         SPSC Ring Buffer │ Heartbeat │ Attestation        │
├─────────────────────────────────────────────────────────────┤
│                      AxonOS Kernel (Cortex-M4F)             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ EDF Scheduler│  │ Signal Pipe │  │ Capability Model │  │
│  │  (U≤0.25)   │  │ (640µs WCET)│  │  (Theorem 8.3)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Consent FSM │  │  Interlock  │  │  Attestation     │  │
│  │  (DC5)      │  │  (Safe-idle)│  │  (ATECC608B)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Hardware Abstraction                    │
│  ADS1299 (ADC) │ nRF52840 (BLE) │ ISO7741 (Isolation)     │
└─────────────────────────────────────────────────────────────┘
```

## Schedulability Guarantees

| Metric | Value | Evidence |
|--------|-------|----------|
| Admission ceiling | U_max = 0.25 | [L1] |
| Binding utilisation | U^L2 = 0.2181 | [L2] |
| WCET (signal pipeline) | C_1^L2 = 818 µs | [L2] |
| Busy period | L = 972 µs | [L2] |
| Deadline slack | S_1 = 3028 µs | [L1] |
| EDF jitter σ | 2.1 µs | [L2] |
| Zero deadline misses | 0 / 10.8×10^6 epochs | [L2] |

## Safety Properties

- **Theorem 6.3**: SPSC sequence-number correctness (Release-Acquire)
- **Theorem 8.3**: Structural data minimisation
- **Theorem 9.1**: Mutual information bound ≤ 140.85 bits/s
- **Theorem 9.3**: Min-entropy residual ≥ H_∞(X) - 7.49 bits

## Kani Verification

| Proof | Property | Unwind | Time |
|-------|----------|--------|------|
| K1 | No data race | 8 | 4.2s |
| K2 | Wait-freedom | 4 | 1.1s |
| K3 | Memory ordering | 2 | 0.8s |
| K4 | Overrun detection | 8 | 2.0s |
| K5 | Consent terminal | 12 | 2.3s |
| K6 | Consent liveness | 12 | 1.8s |
| K7 | Suspended permissions | 12 | 1.5s |

## License

Dual-licensed under Apache-2.0 or MIT at your option.

## Citation

```bibtex
@article{yermakou2026axonos,
  title={AxonOS: Analytical Real-Time Schedulability, Structural Capability Isolation,
         and Empirical Validation of a Safety-Critical Brain Computer Interface Microkernel},
  author={Yermakou, Denis},
  journal={arXiv preprint},
  year={2026}
}
```
