"""Install a built wheel in an isolated environment and check its CLI entry point."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main():
    wheels = list(Path(sys.argv[1]).resolve().glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Expected exactly one wheel")
    # Keep temporary artifacts inside the build directory on all platforms.
    with tempfile.TemporaryDirectory(
        prefix="install-check-", dir=wheels[0].parent
    ) as tmp:
        root = Path(tmp)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        executable_dir = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = executable_dir / ("python.exe" if os.name == "nt" else "python")
        cli = executable_dir / ("system-one.exe" if os.name == "nt" else "system-one")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheels[0])],
            cwd=root,
            env=env,
            check=True,
        )
        help_result = subprocess.run(
            [str(cli), "--help"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        for option in ("--provider", "--model", "--json"):
            assert option in help_result.stdout, help_result.stdout
        invalid = subprocess.run(
            [str(cli), "choice", "state", "instructions"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert invalid.returncode == 2, invalid
        assert "explicit option/level" in invalid.stderr, invalid.stderr
        assert invalid.stdout == "", invalid.stdout
        print("Wheel installation and CLI smoke checks passed")


if __name__ == "__main__":
    main()
