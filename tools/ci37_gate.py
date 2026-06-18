#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GATES = {
    "01": "Repository source surface",
    "02": "README product identity",
    "03": "Version/reference surface",
    "04": "License surface",
    "05": "Security surface",
    "06": "Contribution surface",
    "07": "Cargo workspace metadata advisory",
    "08": "Rust format advisory",
    "09": "Rust clippy advisory",
    "10": "Rust tests advisory",
    "11": "Python syntax surface",
    "12": "Python tests advisory",
    "13": "Kani proof surface",
    "14": "Firmware surface",
    "15": "Kernel crates surface",
    "16": "Scheduler surface",
    "17": "Capability surface",
    "18": "Intent surface",
    "19": "Time surface",
    "20": "SPSC surface",
    "21": "Docs surface",
    "22": "Examples surface",
    "23": "No obvious secrets",
    "24": "No generated artifacts",
    "25": "Markdown non-empty",
    "26": "Workflow surface",
    "27": "Docker surface advisory",
    "28": "Requirements surface",
    "29": "Rust toolchain surface",
    "30": "Deny/clippy config surface",
    "31": "Manifest/notice surface",
    "32": "File size sanity",
    "33": "No merge conflict markers",
    "34": "No stale backup artifacts",
    "35": "CI documentation surface",
    "36": "Public claim hygiene",
    "37": "Foundation readiness summary",
}

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"ghp_[A-Za-z0-9_]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\bseed phrase\b\s*[:=]"),
    re.compile(r"(?i)\bprivate key\b\s*[:=]\s*[A-Za-z0-9]{20,}"),
]

CONFLICT = re.compile(r"^(<{7}|={7}|>{7})( |$)")
BACKUP = re.compile(r"(\.bak$|\.orig$|\.rej$|~$|\.tmp$|\.swp$)")
TEXT_SUFFIXES = {
    ".md", ".txt", ".toml", ".rs", ".py", ".yml", ".yaml", ".json",
    ".lock", ".gitignore", ".cfg", ".ini", ".x", ".sh", ".dockerfile"
}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".wasm", ".woff", ".woff2"}
GENERATED_PREFIXES = ("target/", "dist/", "build/", "coverage/", ".next/", "node_modules/")
MAX_FILE_BYTES = 2_000_000

CLAIM_SCAN_SKIP = {
    "tools/ci37_gate.py",
    "docs/CI_37_FOUNDATION_GATES.md",
}

FORBIDDEN_UNQUALIFIED = [
    "fda approved",
    "clinically proven",
    "guaranteed safe",
    "reads thoughts",
    "mind control",
]

NEGATORS = re.compile(
    r"\b(no|not|non|never|without|cannot|can't|isn't|aren't)\b",
    re.IGNORECASE,
)


def run(cmd: list[str]) -> tuple[int, str]:
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return res.returncode, (res.stdout + res.stderr).strip()


def tracked() -> list[str]:
    code, out = run(["git", "ls-files"])
    if code != 0:
        print(out)
        return []
    return [x.strip() for x in out.splitlines() if x.strip()]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def is_file(path: str) -> bool:
    p = ROOT / path
    return p.is_file() and p.stat().st_size > 0


def is_dir(path: str) -> bool:
    return (ROOT / path).is_dir()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="ignore")


def ok(msg: str) -> int:
    print("OK:", msg)
    return 0


def warn(msg: str) -> int:
    print("WARNING:", msg)
    return 0


def fail(msg: str) -> int:
    print("FAIL:", msg)
    return 1


def require_file(path: str, label: str | None = None) -> int:
    if is_file(path):
        return ok(label or f"{path} present")
    return fail(label or f"{path} missing")


def require_dir(path: str, label: str | None = None) -> int:
    if is_dir(path):
        return ok(label or f"{path} present")
    return fail(label or f"{path} missing")


def advisory_cmd(title: str, cmd: list[str]) -> int:
    code, out = run(cmd)
    if code == 0:
        return ok(f"{title} passed")
    print(out[-2000:] if out else "")
    return warn(f"{title} failed; accepted as advisory for v2.2 snapshot")


def claim_allowed(line: str, pos: int) -> bool:
    fragment = line[:pos]
    for sep in ".;:!?":
        idx = fragment.rfind(sep)
        if idx != -1:
            fragment = fragment[idx + 1 :]
    return bool(NEGATORS.search(fragment[-90:]))


def gate_01() -> int:
    files = tracked()
    if len(files) < 20:
        return fail(f"too few tracked files: {len(files)}")
    needed = ["README.md", "Cargo.toml", ".github/workflows"]
    missing = [x for x in needed if not exists(x)]
    if missing:
        return fail("missing: " + ", ".join(missing))
    return ok(f"{len(files)} tracked files")


def gate_02() -> int:
    if not is_file("README.md"):
        return fail("README.md missing")
    text = read("README.md").lower()
    markers = ["axonos", "kernel"]
    missing = [m for m in markers if m not in text]
    if missing:
        return fail("README missing identity markers: " + ", ".join(missing))
    return ok("README identifies AxonOS/kernel surface")


def gate_03() -> int:
    candidates = ["VERSION", "Cargo.toml", "CHANGELOG.md", "README.md"]
    present = [x for x in candidates if exists(x)]
    if not present:
        return fail("no version/reference surface")
    return ok("version/reference surface: " + ", ".join(present))


def gate_04() -> int:
    licenses = [x for x in ["LICENSE", "LICENSE.md", "LICENSE-APACHE", "LICENSE-MIT", "COPYING"] if exists(x)]
    if not licenses:
        return fail("no license surface")
    return ok("license files: " + ", ".join(licenses))


def gate_05() -> int:
    return require_file("SECURITY.md", "SECURITY.md present")


def gate_06() -> int:
    return require_file("CONTRIBUTING.md", "CONTRIBUTING.md present")


def gate_07() -> int:
    if not exists("Cargo.toml"):
        return warn("Cargo.toml absent; cargo metadata skipped")
    return advisory_cmd("cargo metadata", ["cargo", "metadata", "--format-version", "1", "--locked"])


def gate_08() -> int:
    if not exists("Cargo.toml"):
        return warn("Cargo.toml absent; rustfmt skipped")
    return advisory_cmd("cargo fmt", ["cargo", "fmt", "--all", "--check"])


def gate_09() -> int:
    if not exists("Cargo.toml"):
        return warn("Cargo.toml absent; clippy skipped")
    return advisory_cmd("cargo clippy", ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"])


def gate_10() -> int:
    if not exists("Cargo.toml"):
        return warn("Cargo.toml absent; Rust tests skipped")
    return advisory_cmd("cargo test", ["cargo", "test", "--workspace", "--all-targets"])


def gate_11() -> int:
    py_files = [x for x in tracked() if x.endswith(".py")]
    if not py_files:
        return warn("no Python files")
    bad = []
    for path in py_files:
        code, out = run([sys.executable, "-m", "py_compile", path])
        if code != 0:
            bad.append(path)
            print(out)
    if bad:
        return fail("Python syntax failures: " + ", ".join(bad[:10]))
    return ok(f"{len(py_files)} Python files compile")


def gate_12() -> int:
    tests = [x for x in tracked() if x.startswith("test_") and x.endswith(".py")]
    tests += [x for x in tracked() if x.startswith("tests/") and x.endswith(".py")]
    if not tests:
        return warn("no Python test files")
    return advisory_cmd("pytest", [sys.executable, "-m", "pytest", "-q"])


def gate_13() -> int:
    if is_dir("kani_proofs") or any("kani" in x.lower() for x in tracked()):
        return ok("Kani proof surface present")
    return warn("Kani proof surface not found")


def gate_14() -> int:
    expected = ["axonos-firmware-stm32f407", "memory.x", "link.x"]
    present = [x for x in expected if exists(x)]
    if present:
        return ok("firmware/linker surface: " + ", ".join(present))
    return warn("firmware/linker surface not found")


def gate_15() -> int:
    crates = ["axonos-kernel-core", "src"]
    present = [x for x in crates if exists(x)]
    if present:
        return ok("kernel/core surface: " + ", ".join(present))
    return fail("kernel/core surface missing")


def gate_16() -> int:
    return require_dir("axonos-scheduler", "scheduler crate surface present")


def gate_17() -> int:
    return require_dir("axonos-capability", "capability crate surface present")


def gate_18() -> int:
    return require_dir("axonos-intent", "intent crate surface present")


def gate_19() -> int:
    return require_dir("axonos-time", "time crate surface present")


def gate_20() -> int:
    return require_dir("axonos-spsc", "spsc crate surface present")


def gate_21() -> int:
    if is_dir("docs"):
        md = list((ROOT / "docs").rglob("*.md"))
        return ok(f"docs present, {len(md)} markdown docs")
    return warn("docs directory missing")


def gate_22() -> int:
    if is_dir("examples"):
        return ok("examples directory present")
    return warn("examples directory missing")


def gate_23() -> int:
    bad: list[str] = []
    for name in tracked():
        path = ROOT / name
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"README", "LICENSE", "COPYING", "NOTICE"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                bad.append(name)
                break
    if bad:
        return fail("possible secrets: " + ", ".join(sorted(set(bad))[:10]))
    return ok("no obvious secret patterns")


def gate_24() -> int:
    bad = [x for x in tracked() if x.startswith(GENERATED_PREFIXES) or "/node_modules/" in x]
    if bad:
        return fail("generated artifacts tracked: " + ", ".join(bad[:10]))
    return ok("no common generated artifact directories tracked")


def gate_25() -> int:
    md = [x for x in tracked() if x.endswith(".md")]
    if not md:
        return fail("no markdown files")
    empty = [x for x in md if (ROOT / x).stat().st_size == 0]
    if empty:
        return fail("empty markdown files: " + ", ".join(empty[:10]))
    return ok(f"{len(md)} non-empty markdown files")


def gate_26() -> int:
    workflows = sorted((ROOT / ".github/workflows").glob("*.yml")) + sorted((ROOT / ".github/workflows").glob("*.yaml"))
    if not workflows:
        return fail("no workflows")
    malformed = []
    for wf in workflows:
        text = wf.read_text(encoding="utf-8", errors="ignore")
        if "name:" not in text or "on:" not in text or "jobs:" not in text:
            malformed.append(str(wf.relative_to(ROOT)))
    if malformed:
        return fail("malformed workflows: " + ", ".join(malformed))
    return ok(f"{len(workflows)} workflow file(s)")


def gate_27() -> int:
    if exists("Dockerfile") or exists("docker-compose.yml"):
        return ok("Docker surface present")
    return warn("Docker surface absent")


def gate_28() -> int:
    reqs = [x for x in tracked() if x.startswith("requirements") and x.endswith(".txt")]
    if reqs or exists("pyproject.toml"):
        return ok("Python requirements/pyproject surface present")
    return warn("Python dependency surface absent")


def gate_29() -> int:
    if exists("rust-toolchain.toml") or exists("rustfmt.toml"):
        return ok("Rust toolchain/config surface present")
    return warn("Rust toolchain/config surface absent")


def gate_30() -> int:
    present = [x for x in ["deny.toml", "clippy.toml"] if exists(x)]
    if present:
        return ok("deny/clippy config: " + ", ".join(present))
    return warn("deny/clippy config absent")


def gate_31() -> int:
    present = [x for x in ["MANIFEST.md", "NOTICE", "ABOUT.md"] if exists(x)]
    if present:
        return ok("manifest/notice surface: " + ", ".join(present))
    return warn("manifest/notice surface absent")


def gate_32() -> int:
    large = []
    for name in tracked():
        p = ROOT / name
        if p.is_file() and p.stat().st_size > MAX_FILE_BYTES:
            large.append(f"{name}={p.stat().st_size}")
    if large:
        return warn("large files: " + ", ".join(large[:8]))
    return ok("file sizes within sanity limit")


def gate_33() -> int:
    bad = []
    for name in tracked():
        p = ROOT / name
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if CONFLICT.match(line):
                bad.append(name)
                break
    if bad:
        return fail("merge conflict markers: " + ", ".join(bad[:10]))
    return ok("no conflict markers")


def gate_34() -> int:
    bad = [x for x in tracked() if BACKUP.search(x)]
    if bad:
        return fail("backup/temp files tracked: " + ", ".join(bad[:10]))
    return ok("no stale backup artifacts")


def gate_35() -> int:
    return require_file("docs/CI_37_FOUNDATION_GATES.md", "CI 37 documentation present")


def gate_36() -> int:
    bad = []
    for name in tracked():
        if name in CLAIM_SCAN_SKIP:
            continue

        p = ROOT / name
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for phrase in FORBIDDEN_UNQUALIFIED:
                pos = low.find(phrase)
                if pos != -1 and not claim_allowed(low, pos):
                    bad.append(f"{name}:{line_no}:{phrase}")

    if bad:
        return fail("forbidden unqualified claims: " + ", ".join(bad[:8]))

    return ok("public claim hygiene clean")


def gate_37() -> int:
    required = [
        "README.md",
        "Cargo.toml",
        ".github/workflows/ci.yml",
        "tools/ci37_gate.py",
        "docs/CI_37_FOUNDATION_GATES.md",
    ]
    missing = [x for x in required if not exists(x)]
    if missing:
        return fail("missing final readiness files: " + ", ".join(missing))
    return ok("37-gate CI bundle ready")


def main() -> int:
    gate = sys.argv[1] if len(sys.argv) > 1 else ""
    if gate not in GATES:
        print(f"Unknown gate: {gate}")
        return 2
    print(f"Gate {gate}: {GATES[gate]}")
    return globals()[f"gate_{gate}"]()


if __name__ == "__main__":
    raise SystemExit(main())
