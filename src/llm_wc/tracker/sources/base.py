"""Source adapter interface and shared fetch helpers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from llm_wc.tracker.paths import CACHE, ROOT
from llm_wc.tracker.snapshot import Snapshot


class SourceUnavailable(RuntimeError):
    """The source cannot be fetched here (missing secret, package, or network)."""


def _github_token() -> str | None:
    """GITHUB_TOKEN / GH_TOKEN, else the GitHub CLI's login (local runs), else None."""
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        return token
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


@dataclass
class Context:
    cache: Path = CACHE
    github_token: str | None = field(default_factory=_github_token)
    timeout: int = 60


@dataclass
class Revision:
    """Upstream state. ``revision`` identifies it (commit SHA, dataset SHA, or content hash)."""

    revision: str
    date: str | None = None
    url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "date": self.date, "url": self.url, **self.extra}


class Source:
    """One upstream leaderboard.

    ``latest_revision`` is a cheap check. ``fetch`` downloads and normalizes
    the data at that revision. ``reference_snapshot`` builds the same snapshot
    from the pinned reference inputs under ``resources/`` (for the fidelity
    tests); sources without a reference setting return None.
    """

    key: str = ""
    label: str = ""
    requires_env: tuple[str, ...] = ()
    reference_snapshot_id: str | None = None  # snapshot ID of the pinned reference input

    def latest_revision(self, ctx: Context) -> Revision | None:
        raise NotImplementedError

    def fetch(self, ctx: Context, revision: Revision | None) -> Snapshot:
        raise NotImplementedError

    def reference_snapshot(self) -> Snapshot | None:
        return None

    def check_env(self) -> None:
        load_env_file()
        missing = [v for v in self.requires_env if not os.getenv(v)]
        if missing:
            raise SourceUnavailable(f"needs {', '.join(missing)}")


def load_env_file() -> None:
    """Read KEY=VALUE pairs from the repository's .env (never overrides the environment)."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def http_json(url: str, ctx: Context, github: bool = False) -> Any:
    headers = {"User-Agent": "llm-wc-tracker", "Accept": "application/json"}
    if github:
        headers["Accept"] = "application/vnd.github+json"
        if ctx.github_token:
            headers["Authorization"] = f"Bearer {ctx.github_token}"
    with urlopen(Request(url, headers=headers), timeout=ctx.timeout) as resp:
        return json.loads(resp.read().decode())


def github_latest_commit(repo: str, path: str, ctx: Context) -> Revision:
    """Most recent commit touching ``path`` in ``repo``."""
    url = f"https://api.github.com/repos/{repo}/commits?path={quote(path)}&per_page=1"
    items = http_json(url, ctx, github=True)
    if not items:
        raise SourceUnavailable(f"no commits for {repo}:{path}")
    sha = items[0]["sha"]
    return Revision(sha, items[0]["commit"]["committer"]["date"], f"https://github.com/{repo}/commit/{sha}")


def github_contains(repo: str, base: str, head: str, ctx: Context) -> bool:
    """True when commit ``head`` is already contained in ``base`` (``base`` is at or after it)."""
    if base == head:
        return True
    data = http_json(f"https://api.github.com/repos/{repo}/compare/{head}...{base}", ctx, github=True)
    return data.get("status") in ("ahead", "identical")


def run_script(args: list[str], ctx: Context, env: dict | None = None) -> None:
    """Run a repository script with this interpreter; raise on failure."""
    full_env = {**os.environ, **(env or {})}
    if ctx.github_token:
        full_env.setdefault("GITHUB_TOKEN", ctx.github_token)
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, env=full_env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"{args[0]} failed (exit {proc.returncode}):\n{tail}")


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True)


def ensure_partial_clone(url: str, dest: Path, revision: str | None = None) -> Path:
    """Blobless, no-checkout clone (reused across runs); fetch ``revision`` if given."""
    if not (dest / ".git").exists() and not (dest / "HEAD").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--quiet", url, str(dest)], check=True)
    else:
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "origin"], check=True)
    if revision:
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "--depth=1", "origin", revision], check=False)
    return dest


def short_date(iso: str | None) -> str:
    return (iso or "")[:10].replace("-", "")
