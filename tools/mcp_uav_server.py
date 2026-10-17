"""
FastMCP server for t2sem-mec-v2 (UAV semantic MEC): experiment status, configs, results.
Run: python tools/mcp_uav_server.py
"""
import json
import subprocess
import sys
from pathlib import Path

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PAPER_DIR = PROJECT_ROOT / "paper"

mcp = FastMCP("uav-t2sem-mec")


def _safe_child(base: Path, name: str) -> Path | None:
    """Resolve a flat file name inside base; reject separators and traversal."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    candidate = (base / name).resolve()
    if candidate.parent != base.resolve():
        return None
    return candidate if candidate.is_file() else None


def _read_limited(path: Path, limit: int) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > limit:
        content = content[:limit] + "\n... [truncated]"
    return content


@mcp.tool()
def project_status() -> str:
    """Overall project overview: configs, scripts, src modules, results, paper files."""
    def names(d: Path, pattern: str, cap: int = 30):
        if not d.exists():
            return []
        return sorted(f.name for f in d.glob(pattern))[:cap]

    return json.dumps({
        "project_root": str(PROJECT_ROOT),
        "configs": names(CONFIG_DIR, "*.yaml"),
        "experiments": names(EXPERIMENTS_DIR, "*.py"),
        "scripts": names(SCRIPTS_DIR, "*.py"),
        "src_modules": sorted(
            f.relative_to(SRC_DIR).as_posix()
            for f in SRC_DIR.rglob("*.py")
            if "__pycache__" not in f.parts
        )[:30] if SRC_DIR.exists() else [],
        "results_files": len([f for f in RESULTS_DIR.rglob("*") if f.is_file()]) if RESULTS_DIR.exists() else 0,
        "paper_files": names(PAPER_DIR, "*", 20),
        "experiment_chain_doc": str(PROJECT_ROOT.parent / "EXPERIMENTS_IN_LOGICAL_ORDER.md"),
    }, indent=2)


@mcp.tool()
def list_configs() -> str:
    """List all experiment config YAML files in configs/"""
    if not CONFIG_DIR.exists():
        return json.dumps({"error": "configs/ not found"})
    configs = sorted(f.name for f in CONFIG_DIR.glob("*.yaml"))
    return json.dumps({"configs": configs, "count": len(configs)})


@mcp.tool()
def read_config(name: str) -> str:
    """Read a config YAML file. Provide name with .yaml extension."""
    path = _safe_child(CONFIG_DIR, name)
    if path is None:
        available = sorted(f.name for f in CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.exists() else []
        return json.dumps({"error": f"Not found: {name}", "available": available})
    return _read_limited(path, 16000)


@mcp.tool()
def list_results(limit: int = 50) -> str:
    """List result files (newest first) with size and modified time."""
    if not RESULTS_DIR.exists():
        return json.dumps({"error": "results/ not found"})
    files = [f for f in RESULTS_DIR.rglob("*") if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    cap = max(1, min(int(limit), 200))
    out = [{
        "path": f.relative_to(RESULTS_DIR).as_posix(),
        "size_kb": round(f.stat().st_size / 1024, 1),
        "mtime": f.stat().st_mtime,
    } for f in files[:cap]]
    return json.dumps({"results": out, "total": len(files), "shown": len(out)}, indent=2)


@mcp.tool()
def read_result(name: str, max_chars: int = 8000) -> str:
    """Read a results/ file by relative name (e.g. 'final/metrics.json'). Subpaths allowed inside results/."""
    if not name or ".." in name:
        return json.dumps({"error": f"Invalid name: {name}"})
    candidate = (RESULTS_DIR / name).resolve() if RESULTS_DIR.exists() else None
    if candidate is None or RESULTS_DIR.resolve() not in candidate.parents or not candidate.is_file():
        return json.dumps({"error": f"Not found: {name}"})
    return _read_limited(candidate, max(1000, min(int(max_chars), 40000)))


@mcp.tool()
def run_smoke(timeout_s: int = 300) -> str:
    """Run sanity + smoke tests (tests/test_sanity.py, experiments/run_smoke.py).
    Uses this server's Python interpreter; if project deps are missing here, run the smoke in the project venv manually."""
    timeout = max(30, min(int(timeout_s), 1200))
    steps = [["tests/test_sanity.py"], ["experiments/run_smoke.py"]]
    reports = []
    for step in steps:
        try:
            r = subprocess.run(
                [sys.executable, *step],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            reports.append({
                "step": step[0],
                "returncode": r.returncode,
                "stdout_tail": r.stdout[-2000:],
                "stderr_tail": r.stderr[-1000:] if r.returncode != 0 else "",
            })
            if r.returncode != 0:
                break
        except subprocess.TimeoutExpired:
            reports.append({"step": step[0], "error": f"Timeout (>{timeout}s)"})
            break
        except Exception as e:  # noqa: BLE001
            reports.append({"step": step[0], "error": str(e)})
            break
    return json.dumps({"reports": reports}, indent=2)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        print(project_status())
    else:
        mcp.run()
