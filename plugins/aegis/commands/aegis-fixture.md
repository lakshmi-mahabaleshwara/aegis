---
description: Generate a synthetic medical-image fixture (fake burnt-in text + fake header PHI) for testing (runs the aegis-fixture CLI)
argument-hint: <output_path.dcm|.png|.jpg> [--text "LINE" ...]
allowed-tools: Bash
---

Generate a synthetic, PHI-free test fixture with the Aegis fixture CLI.

Arguments given: **`$ARGUMENTS`**

Execute:

```bash
"${AEGIS_ROOT:-/Users/lakshmi/aegis}/venv/bin/aegis-fixture" $ARGUMENTS
```

The output is entirely synthetic (no real patient data), so it is safe to
create anywhere and to inspect. Report the output path and the burnt-in text
lines that were baked in. This produces *test input only* — it is not
de-identification.

If `$ARGUMENTS` is empty, ask the user for an output path and (optionally) the
burnt-in `--text` lines. A good default for a demo:
`aegis-fixture /tmp/demo.dcm --text "SYNTHETIC PATIENT" --text "ID SYN-0001"`.
