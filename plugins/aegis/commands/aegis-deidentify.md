---
description: De-identify a DICOM/JPEG/PNG file or directory, emitting a PHI-free JSON envelope (runs the aegis-deidentify CLI)
argument-hint: <input_path> --output-dir <dir>
allowed-tools: Bash
---

De-identify a file or directory with the Aegis CLI.

Arguments given: **`$ARGUMENTS`**

Execute (stderr is logs; the PHI-free JSON envelope is on stdout):

```bash
"${AEGIS_ROOT:-/Users/lakshmi/aegis}/venv/bin/aegis-deidentify" $ARGUMENTS 2>/dev/null
```

Then report from the JSON envelope: `status`, files processed, header tags
scrubbed, pixel decisions (redacted / safelisted / low_confidence), and whether
anything `needs_manual_review`. The envelope contains counts only — never echo
verbatim OCR text or file contents.

**PHI safety:** do not read the input files' contents. If the input is under
`staging_input/` or another real-data directory, refer to files by name only.

If `$ARGUMENTS` is missing an `--output-dir`, ask the user where outputs should
go before running.
