"""
Command-line interface for System One decisions.
"""

from __future__ import annotations
import sys
from .client import SystemOneClient


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: system-one <type: noul|choice|score> <state/text> [instructions] [options...]"
        )
        print("Examples:")
        print(
            "  system-one noul 'User reported payout failure' 'Is this an urgent production bug?'"
        )
        print(
            "  system-one choice 'Payment gateway 500 error' 'Department' Backend DevOps Support"
        )
        print(
            "  system-one score 'Critical outage detected' 'Severity' Low Medium High Critical"
        )
        sys.exit(1)

    q_type = sys.argv[1].lower()
    state = sys.argv[2]
    instr = sys.argv[3] if len(sys.argv) > 3 else "Evaluate the provided state"

    client = SystemOneClient()

    if q_type == "noul":
        prob = client.noul(state, instr)
        print(f"Probability (True/Yes): {prob:.2f}")
    elif q_type == "choice":
        options = sys.argv[4:] if len(sys.argv) > 4 else ["Option_A", "Option_B"]
        selected = client.choice(state, instr, options)
        print(f"Selected: {selected}")
    elif q_type == "score":
        levels = sys.argv[4:] if len(sys.argv) > 4 else ["Low", "Medium", "High"]
        lvl = client.score(state, instr, levels)
        print(f"Score Level: {lvl}")
    else:
        print(f"Unknown question type: {q_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
