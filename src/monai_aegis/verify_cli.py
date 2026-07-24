"""``aegis-verify`` — checklist verification of a de-identified run directory.

The independent second pass over Aegis output: re-opens the artifacts and
reports and asserts the de-identification claims, emitting a PHI-free JSON
report on stdout (schema: ``monai_aegis/schemas/verification.schema.json``).
Which assertions run is defined by a YAML checklist, not code — see
``monai_aegis/checklists/ps315.yaml`` for the packaged default.

Contract:
  - stdout carries exactly one JSON report; all logging goes to stderr.
  - Exit codes:
      0  pass — no error-severity findings (warnings may be present)
      1  verification could not run (unreadable file, invalid checklist)
      2  invalid invocation — run directory does not exist
      3  fail — one or more error-severity checks failed
    A report or error document is emitted on every exit code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_INVALID_INPUT = 2
EXIT_FAIL = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis-verify",
        description=(
            "Verify a de-identified Aegis run directory against a checklist, "
            "emitting a PHI-free JSON report on stdout. Assertions are "
            "config-driven: customize via --checklist YAML, never flags."
        ),
    )
    parser.add_argument("run_dir", help="De-identified run directory to verify.")
    parser.add_argument(
        "--checklist",
        default=None,
        help="Checklist YAML (default: the packaged PS3.15 checklist).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent the JSON report (still a single stdout document).",
    )
    return parser


def _emit(document: dict, pretty: bool) -> None:
    json.dump(document, sys.stdout, indent=2 if pretty else None, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="[aegis-verify] %(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    from monai_aegis import verify
    from monai_aegis.api import InputError

    try:
        report = verify.verify_run(args.run_dir, checklist=args.checklist)
    except InputError as exc:
        _emit({"status": "error", "message": str(exc), "run_dir": args.run_dir}, args.pretty)
        return EXIT_INVALID_INPUT
    except Exception as exc:  # invalid checklist, unreadable DICOM, ...
        logging.getLogger("aegis-verify").exception("Verification failed to run")
        _emit(
            {"status": "error", "message": f"{type(exc).__name__}: {exc}", "run_dir": args.run_dir},
            args.pretty,
        )
        return EXIT_ERROR

    _emit(report, args.pretty)
    return EXIT_PASS if report["status"] == verify.STATUS_PASS else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
