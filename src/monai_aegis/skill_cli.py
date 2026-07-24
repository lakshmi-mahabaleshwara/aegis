"""``aegis-deidentify`` — single-invocation skill CLI.

The scriptable surface of Aegis for agent-skill harnesses: one command in,
one JSON envelope out. Behavior is configured exclusively through YAML
(base config + overlay + env interpolation) — the flags here identify the
invocation (what to process, where to write, which config), never override
behavior.

Contract:
  - stdout carries exactly one JSON envelope (see
    ``monai_aegis/schemas/envelope.schema.json``); all logging goes to
    stderr. Logging is configured before monai_aegis imports run, so
    library initialization can never write to stdout.
  - Exit codes:
      0  success — every file processed, nothing needs review
      1  run failed — no file processed (envelope status "error")
      2  invalid invocation — bad input path, unknown mode
      3  partial — some files processed, some failed ("partial")
      4  success, but at least one output needs manual review
    An envelope is emitted on every exit code, including failures.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID_INPUT = 2
EXIT_PARTIAL = 3
EXIT_NEEDS_REVIEW = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis-deidentify",
        description=(
            "De-identify a medical image file (DICOM/JPEG/PNG) or a directory "
            "of them, emitting a PHI-free JSON envelope on stdout. Behavior is "
            "config-driven: customize via --overlay YAML, never flags."
        ),
    )
    parser.add_argument("input", help="File or directory to de-identify.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for de-identified artifacts and reports "
        "(created if missing; no timestamped subdirectories are added).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Base config YAML (default: the packaged monai_aegis config).",
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Overlay YAML deep-merged onto the base config (default: the "
        "packaged skill overlay, unless AEGIS_CONFIG_OVERRIDE is set).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "dicom", "image"],
        default="auto",
        help="Directory discovery mode (ignored for single-file input).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent the JSON envelope (still a single stdout document).",
    )
    return parser


def _emit(envelope_dict: dict, pretty: bool) -> None:
    json.dump(envelope_dict, sys.stdout, indent=2 if pretty else None, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


class _StdoutReserved:
    """Point fd 1 at stderr for the duration of the pipeline run.

    Any stray write to stdout — a ``print()`` in a dependency, a native
    library's progress output — lands in the log instead of corrupting the
    envelope stream. The original fd is restored on exit, *before* the
    envelope is emitted. A no-op when stdout has no real file descriptor
    (e.g. embedded in a test harness capturing at the Python level).
    """

    def __enter__(self):
        self._saved_fd = None
        try:
            sys.stdout.flush()
            self._saved_fd = os.dup(sys.stdout.fileno())
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
            self._saved_fd = None
        return self

    def __exit__(self, *exc_info):
        if self._saved_fd is not None:
            os.dup2(self._saved_fd, sys.stdout.fileno())
            os.close(self._saved_fd)
        return False


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # stderr-only logging, configured before any monai_aegis import can
    # initialize a library that logs — stdout stays envelope-only.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="[aegis-deidentify] %(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    from monai_aegis import api, envelope

    exit_code = EXIT_SUCCESS
    try:
        with _StdoutReserved():
            result = api.deidentify(
                input_path=args.input,
                output_dir=args.output_dir,
                config_path=args.config,
                overlay_path=args.overlay,
                mode=args.mode,
            )
    except api.InputError as exc:
        result = envelope.error_envelope(str(exc), args.input, args.output_dir)
        exit_code = EXIT_INVALID_INPUT
    except Exception as exc:  # config errors, model failures before any file
        logging.getLogger("aegis-deidentify").exception("Run failed")
        result = envelope.error_envelope(f"{type(exc).__name__}: {exc}", args.input, args.output_dir)
        exit_code = EXIT_ERROR
    else:
        if result["status"] == envelope.STATUS_ERROR:
            exit_code = EXIT_ERROR
        elif result["status"] == envelope.STATUS_PARTIAL:
            exit_code = EXIT_PARTIAL
        elif result.get("needs_manual_review"):
            exit_code = EXIT_NEEDS_REVIEW

    _emit(result, args.pretty)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
