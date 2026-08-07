"""Checks for the maintained, Agent-facing documentation surface."""

import re
import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ACTIVE_DOCS = [
    REPO / "README.md",
    REPO / "scripts/README.md",
    REPO / "configs/README.md",
    REPO / "runs/README.md",
    REPO / "runs/FILE_MANAGEMENT_GUIDELINES.md",
    *sorted((REPO / "skill").glob("*.md")),
]


def test_required_agent_entry_points_exist():
    for path in (
        REPO / "README.md",
        REPO / "skill/SKILL.md",
        REPO / "skill/SKILLS_INDEX.md",
        REPO / "scripts/README.md",
        REPO / "pyproject.toml",
    ):
        assert path.is_file(), f"missing documentation entry point: {path}"


def test_package_version_matches_runtime_version():
    metadata = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = (REPO / "squid/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', runtime, re.MULTILINE)
    assert match, "squid.__version__ is missing"
    assert metadata["project"]["version"] == match.group(1)


def test_active_docs_are_ascii_and_local_links_resolve():
    link_re = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    for doc in ACTIVE_DOCS:
        payload = doc.read_bytes()
        assert payload.isascii(), f"non-ASCII text in active doc: {doc}"
        text = payload.decode("ascii")
        for target in link_re.findall(text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (doc.parent / target).resolve().exists(), (
                f"broken local link in {doc}: {target}"
            )


def test_documented_script_paths_exist():
    command_re = re.compile(
        r"(?:python|\$PY)\s+(scripts/[A-Za-z0-9_./-]+\.py)"
    )
    commands = {
        command
        for doc in ACTIVE_DOCS
        for command in command_re.findall(doc.read_text(encoding="ascii"))
    }
    assert commands, "no script commands found in active documentation"
    missing = sorted(command for command in commands if not (REPO / command).is_file())
    assert not missing, f"documented scripts do not exist: {missing}"


def test_repository_root_has_no_generated_products():
    patterns = (
        "wout_*.nc",
        "input.*",
        "threed1.*",
        "simsopt_*.dat",
        "parvmecinfo.txt",
        "boozmn_*.nc",
    )
    hits = sorted(
        path
        for pattern in patterns
        for path in REPO.glob(pattern)
        if path.is_file()
    )
    assert not hits, f"generated products found in repository root: {hits}"
