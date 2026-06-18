# CI 37 Foundation Gates

This repository uses a 37-job CI matrix for archival and foundation-grade repository validation.

The workflow validates:
- repository structure,
- README and documentation surface,
- license/security/contribution files,
- Rust workspace surface,
- Python surface,
- firmware/linker surface,
- Kani proof surface,
- no obvious secret leaks,
- no generated artifacts,
- workflow presence,
- final foundation readiness.

Some Rust/Python build checks are advisory because this repository is a historical v2.2 snapshot with mixed Rust, firmware, Python, and proof assets. Advisory checks report warnings but do not block archival validation.
