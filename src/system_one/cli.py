"""Command-line interface for validated System One decisions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .client import SystemOneClient
from .errors import InvalidResponseError, ProviderError
from .primitives import Choice, Noul, Score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="system-one",
        description="Evaluate a structured decision. Choice/score require explicit options/levels.",
    )
    parser.add_argument("type", choices=("noul", "choice", "score"))
    parser.add_argument("state", help="Text to evaluate")
    parser.add_argument(
        "instructions", nargs="?", default="Evaluate the provided state"
    )
    parser.add_argument(
        "options", nargs="*", help="Allowed choices or ordered score levels"
    )
    parser.add_argument("--provider", choices=("gemini", "groq", "openai", "ollama"))
    parser.add_argument("--model", help="Override the provider's default model")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print answers and metrics as JSON; raw provider data is omitted",
    )
    args = parser.parse_intermixed_args(argv)
    if args.type in ("choice", "score") and not args.options:
        parser.error(
            f"{args.type} requires instructions and at least one explicit option/level"
        )
    if args.type == "noul" and args.options:
        parser.error("noul does not accept options/levels")
    try:
        if args.type == "choice":
            question = Choice(options=args.options, instructions=args.instructions)
        elif args.type == "score":
            question = Score(levels=args.options, instructions=args.instructions)
        else:
            question = Noul(instructions=args.instructions)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        client = SystemOneClient(provider=args.provider, model=args.model)
        try:
            result = client.evaluate(args.state, {"q": question})
        finally:
            client.close()
    except (InvalidResponseError, ProviderError, ValueError) as exc:
        if args.json_output:
            print(
                json.dumps(
                    {"error": {"type": type(exc).__name__, "message": str(exc)}},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        else:
            print(f"system-one: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(
            json.dumps(
                {
                    "answers": {
                        key: asdict(answer) for key, answer in result.answers.items()
                    },
                    "metrics": asdict(result.metrics),
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    else:
        value = result.answers["q"].value
        if args.type == "noul":
            print(f"Probability (True/Yes): {value:.2f}")
        elif args.type == "choice":
            print(f"Selected: {value}")
        else:
            print(f"Score Level: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
