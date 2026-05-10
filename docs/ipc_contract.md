# IPC Contract (DC1-DC6)

## Dual-Core Partition

- **M4F**: Hard real-time (signal pipeline, consent, interlock)
- **A53**: Soft real-time (session, BLE, WASM sandbox)

## Clauses

| Clause | Description | Violation Response |
|--------|-------------|-------------------|
| DC1 | Pipeline deadline ≤ 4 ms | Safe-idle |
| DC2 | Intent packet integrity | Drop packet |
| DC3 | Capability isolation | Kill app |
| DC4 | Mutual information ≤ 140.85 bits/s | Throttle |
| DC5 | Heartbeat ≤ 12 ms | Safe-idle |
| DC6 | HMAC attestation | Reject intent |

## References

- AxonOS RFC-0006: Dual-Core Contract Specification.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
