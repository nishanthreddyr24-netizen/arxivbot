"""Run the evaluation unattended, for as long as it is given.

Launch this before bed. It alternates between the two stages, keeps going when
the provider refuses a burst, waits out a daily quota rather than dying on it,
and checkpoints after every paper so a night that ends early still leaves
usable results.

    python eval/overnight.py            # runs until the corpus is covered
    python eval/overnight.py 8          # stops after 8 hours

Safe to interrupt and safe to re-run: everything already recorded is skipped.
It only reads papers and writes to eval/, and changes no source file - the
point of an unattended run is that nothing can be quietly broken while nobody
is watching.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "eval" / "run.py"
LOG = ROOT / "eval" / "overnight.log"

# A daily quota resets on a clock, not a timer, so waiting it out beats
# hammering it. A burst limit clears in a minute; this is for the other kind.
QUOTA_WAIT = timedelta(minutes=45)


def say(message: str) -> None:
    stamped = f"[{datetime.now():%H:%M:%S}] {message}"
    print(stamped, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(stamped + "\n")


def stage(name: str) -> tuple[int, str]:
    """One pass of a stage. Returns its exit code and tail of its output."""
    result = subprocess.run(
        [sys.executable, str(RUNNER), name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


def remaining(output: str) -> int:
    """How many papers that pass still had to do, read from its own report."""
    for line in output.splitlines():
        if "to run" in line or "need more runs" in line:
            for token in line.replace(",", " ").split():
                if token.isdigit():
                    return int(token)
    return 0


def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 0
    deadline = datetime.now() + timedelta(hours=hours) if hours else None

    say(f"starting{f' for {hours}h' if hours else ''}")
    idle_rounds = 0

    while True:
        if deadline and datetime.now() >= deadline:
            say("time budget reached")
            break

        progressed = False
        for name in ("breadth", "stability"):
            if deadline and datetime.now() >= deadline:
                break
            code, output = stage(name)
            tail = [l for l in output.splitlines() if l.strip()][-1:] or [""]
            say(f"{name}: exit {code} · {tail[0][:90]}")

            if "quota appears exhausted" in output:
                say(f"quota exhausted; sleeping {QUOTA_WAIT}")
                time.sleep(QUOTA_WAIT.total_seconds())
                progressed = True
                continue
            if remaining(output) > 0 or code != 0:
                progressed = True

        if not progressed:
            idle_rounds += 1
            # Two quiet rounds in a row means both stages report nothing left.
            if idle_rounds >= 2:
                say("nothing left to do — corpus covered")
                break
            time.sleep(30)
        else:
            idle_rounds = 0

    say("writing final report")
    _, output = stage("report")
    print(output)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(output + "\n")
    say("done")


if __name__ == "__main__":
    main()
