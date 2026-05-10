# Validation

## L1 Evidence

- Instruction counts derived from `thumbv7em-none-eabihf` assembly.

## L2 Evidence

- DWT cycle counter measurements on STM32F407 @ 168 MHz.
- 10.8M epochs, zero deadline misses.

## L3 Evidence

- Oscilloscope validation with Saleae Logic Pro 16.

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
