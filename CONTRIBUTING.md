# Contributing to AxonOS

## Development Setup

```bash
rustup target add thumbv7em-none-eabihf thumbv8m.main-none-eabihf
cargo install cargo-kani
cargo install cargo-binutils
```

## Code Standards

- All code must be `#![no_std]` compatible
- Targeted `unsafe` requires a Kani proof or L3 evidence
- Every quantitative claim requires an evidence label [L1/L2/L3]
- All modules must include academic references in doc comments
- Clippy warnings are treated as errors in CI

## Pull Request Process

1. Open an RFC in `axonos-rfcs` for architectural changes
2. Ensure `cargo test --lib` and `cargo kani --features kani` pass
3. Update `docs/` and `CHANGELOG.md`
4. Request review from @denis-yermakou
