# Changelog

## v0.3.7 — 2026-05-11

### Fixed
- **Unsafe scope corrected**: `#![allow(unsafe_code)]` properly scoped to
  `platform/dwt`, `platform/gpio`, `hal`, `ringbuf/spsc`, `consent/interlock`
  instead of global `forbid` conflicting with targeted unsafe
- **SPSC generic type**: `SpscRingBuffer<T>` correctly propagates generic
  through all methods; `const { MaybeUninit::uninit() }` array init fixed
- **EDF WCRT calculation**: `tick()` now safely handles missing parent task
  via `find()` + `unwrap_or(0)` instead of potential panic
- **Dual-core data race**: `DualCoreContract::safe_idle_active` changed from
  `bool` to `AtomicBool` to prevent data race between M4F writer and A53 reader
- **Heartbeat atomicity**: `send_heartbeat()` now atomically clears safe-idle
  flag via `AtomicBool::store(false, Release)`
- **Manifest builder**: `app_id()` now correctly handles `heapless::String`
  capacity via `from_str` + `map_err`
- **Interlock GPIO**: `activate_safe_idle()` is now safe wrapper around
  atomic BSRR write; callable from panic handler without unsafe block

### Changed
- `deny(unsafe_code)` crate-wide with explicit `allow` in 5 modules
- `DualCoreContract::check_heartbeat()` uses `Acquire`/`Release` ordering
  for `safe_idle_active` instead of plain bool
- `EdfScheduler` removed unused `dwt` field (was dead code)

### Added
- `platform/dwt.rs`: `#![allow(unsafe_code)]` for DWT register access
- `platform/gpio.rs`: `#![allow(unsafe_code)]` for memory-mapped GPIO
- `hal/mod.rs`: `#![allow(unsafe_code)]` for `mmio_read`/`mmio_write`
- `consent/interlock.rs`: `#![allow(unsafe_code)]` for GPIO atomic ops

## v0.2.4 — 2026-05-10

### Fixed
- SPSC ring buffer generic over `T`
- EDF scheduler ready queue push/pop
- `heapless::Vec` capacity parameters
- Pipeline silent drop on overrun
- Interlock pure/effect split
- 64-bit DWT timestamps
- SPSC `reset()` drops pending items
- Manifest signature placeholder

## v0.1.0 — 2026-04-01

- Initial release
