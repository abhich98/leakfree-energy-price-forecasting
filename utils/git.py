import subprocess


def _git_output(*args: str, cwd: str | None = None) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def get_git_sha() -> str | None:
    """Return the current commit SHA when the source is inside a Git checkout."""
    try:
        repository_root = _git_output("rev-parse", "--show-toplevel")
        return _git_output("rev-parse", "HEAD", cwd=repository_root)
    except (OSError, subprocess.CalledProcessError):
        return None


def get_git_metadata() -> dict[str, str | bool | None]:
    """Return Git provenance for the current checkout without raising outside Git."""
    try:
        repository_root = _git_output("rev-parse", "--show-toplevel")
        commit_sha = _git_output("rev-parse", "HEAD", cwd=repository_root)
        short_commit_sha = _git_output(
            "rev-parse", "--short", "HEAD", cwd=repository_root
        )
        is_dirty = bool(_git_output("status", "--porcelain", cwd=repository_root))
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}

    try:
        branch = _git_output(
            "symbolic-ref", "--short", "-q", "HEAD", cwd=repository_root
        )
    except subprocess.CalledProcessError:
        branch = None

    return {
        "available": True,
        "commit_sha": commit_sha,
        "short_commit_sha": short_commit_sha,
        "branch": branch or None,
        "is_dirty": is_dirty,
        "repository_root": repository_root,
    }