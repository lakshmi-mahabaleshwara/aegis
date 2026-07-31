---
description: Audit a de-identified Aegis run directory against the PS3.15 checklist (runs the aegis-verify CLI)
argument-hint: <run_dir>
allowed-tools: Bash
---

Run the Aegis checklist verifier on this run directory: **`$ARGUMENTS`**

Execute exactly this (stderr is logs; the PHI-free JSON report is on stdout):

```bash
"${AEGIS_ROOT:-/Users/lakshmi/aegis}/venv/bin/aegis-verify" $ARGUMENTS 2>/dev/null
```

Then report the result: overall `status` (pass/fail), `checks_evaluated`,
`failures`, `warnings`, and list any `findings`. The report is PHI-free by
design, so it is safe to show in full.

If `$ARGUMENTS` is empty, ask the user which run directory to verify (e.g. a
folder under `staging_output/`).
