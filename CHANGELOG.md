# Changelog

## v0.2.4 — 2026-05-10

### Fixed
- SPSC ring buffer is now generic over `T` and wraparound-safe (u32 wrapping arithmetic)
- EDF scheduler ready queue now properly pushes and pops jobs
- `heapless::Vec` capacity parameters added throughout
- Panic handler no longer conflicts with targeted `unsafe` in ringbuf
- Pipeline no longer silently drops classifier output on SPSC overrun
- Interlock split into pure logic (`next_state`) and effectful execution (`apply_state`)
- `DualCoreContract` uses 64-bit atomic timestamps with DWT overflow handling
- SPSC `reset()` now drops pending `Published` items via `drop_in_place`
- Manifest builder documents unsigned signature placeholder (DC6)
- All modules now include academic references and RFC citations

### Added
- `platform::Dwt` with 64-bit virtual cycle counter (wraparound-safe)
- `tests/integration.rs` with wraparound stress test (1M iterations)
- `kani_proofs/consent.rs` with FSM safety, liveness, and permission proofs (K5-K7)
- `scripts/measure_wcrt.py` for EVT fitting of WCRT measurements
- `.github/workflows/ci.yml` with host tests, clippy, cross-build, and Kani
- `docs/references.md` with 21 academic citations
- Comprehensive doc comments with theorem references in all source files

### Changed
- Crate-level `#![forbid(unsafe_code)]` relaxed to `#![deny(unsafe_code)]`
- `unsafe` explicitly allowed only in `ringbuf::spsc` with safety comments
- Consent FSM: `is_processing_allowed()` now returns true for `Suspended` (buffering)
- `busy_period_bound()` uses saturating arithmetic with overflow guards

## v0.2.2 — 2026-05-10

- Intermediate release with compilation fixes

## v0.1.0 — 2026-04-01

- Initial release
