# Aegis Skill Kit — Adapter Contract

Aegis is built to be wrapped. Whatever agent-skill ecosystem you target —
Anthropic/Claude Skills, a manifest-based catalog (e.g. NVIDIA MedTech
Medical AI Skills), an MCP host, a bespoke orchestrator — the wrapper you
write is a thin adapter over one stable surface. This kit documents that
surface so an adapter is a page of glue, not a re-implementation.

- [`SKILL.template.md`](SKILL.template.md) — a fill-in-the-blanks
  agent-skill card you can copy into a catalog.
- [`DISCLAIMER.md`](DISCLAIMER.md) — the reusable PHI-scope and limitations
  text every ecosystem asks for. Copy it verbatim; keep the wording.

## The pattern

Every wrapper, in every ecosystem, does the same three things:

1. **Resolve the invocation** — an input path (file or directory) and an
   output directory. Nothing else is required.
2. **Invoke the facade** — one call, either the `aegis-deidentify` command
   or `monai_aegis.api.deidentify(...)`.
3. **Serialize the envelope** — take the returned PHI-free JSON envelope and
   render it for your host (a tool result, a report card, a log line).

Behavior is never configured through the wrapper. It is configured through
YAML (base config + overlay + environment interpolation). An adapter that
needs different behavior ships a different overlay, not different code.

```text
input path ─▶ aegis-deidentify / api.deidentify ─▶ envelope (PHI-free JSON)
                        │                                     │
                   config + overlay                     serialize for host
                   (YAML, not flags)                    (+ optional verify)
```

## The stable surface

### Commands

| Command | Purpose | Stdout | Exit codes |
|---|---|---|---|
| `aegis-deidentify INPUT --output-dir OUT` | De-identify a file or directory | one envelope JSON (`schemas/envelope.schema.json`) | `0` success · `1` failed · `2` bad invocation · `3` partial · `4` success, files need review |
| `aegis-verify RUN_DIR` | Independently verify a run against a checklist | one report JSON (`schemas/verification.schema.json`) | `0` pass · `1` could not run · `2` bad invocation · `3` checks failed |
| `aegis-fixture OUT.dcm` | Generate a synthetic fixture (no real data) | the written path | `0` ok |

All three log only to stderr; stdout carries exactly one JSON document. The
equivalent library calls are `monai_aegis.api.deidentify(...)`,
`monai_aegis.verify.verify_run(...)`, and
`monai_aegis.fixtures.make_synthetic_dicom(...)`.

### Contracts a wrapper may rely on

- **Caller-owned output.** Artifacts and ground-truth reports are written
  directly into `--output-dir`; no timestamped subdirectories are inserted.
- **PHI-free output.** The envelope, the verification report, and the
  default `aegis_pixel_detections.csv` never contain pixel data, OCR text,
  text tokens, or DICOM tag values — only file names, paths, counts,
  decision categories, and timings. Verbatim OCR text is opt-in
  (`reporting.include_phi_text`) and only ever written outside the output
  directory.
- **Determinism.** Identical input + identical config + identical
  `AEGIS_TOKEN_SALT` reproduce byte-identical artifacts, reports, and
  envelope (modulo timing). Tokens and regenerated DICOM UIDs are both
  deterministic. This is what lets a harness verify by re-running.
- **Fail loud, don't guess.** Invalid input or an unresolvable model fails
  with a clear message and a non-zero exit code and an emitted envelope —
  never a silent partial result.
- **Independent verifiability.** `aegis-verify` re-derives the
  de-identification claim from the output itself, using only pydicom +
  PyYAML — so a verifier can run where the OCR/NER models are not installed.

### Schemas (gate against these, don't re-author them)

- `monai_aegis/schemas/envelope.schema.json` — the run envelope.
- `monai_aegis/schemas/verification.schema.json` — the verification report.

Both are shipped inside the installed package and are also reachable in
code via `monai_aegis.envelope.schema()` and
`monai_aegis.verify.validate_report(...)`.

### Environment variables

| Variable | Purpose |
|---|---|
| `AEGIS_TOKEN_SALT` | Tokenization/UID salt. Fix it for reproducible, re-linkable output. |
| `AEGIS_CONFIG_OVERRIDE` | Path to an overlay YAML; replaces the default skill overlay. |
| `AEGIS_DEVICE` | `cpu` (default), `cuda`, or `mps`. |
| `AEGIS_HF_OFFLINE` | `true` → NER resolves from local cache only (no network). |
| `AEGIS_MODEL_DOWNLOADS` | `false` → EasyOCR never downloads. |
| `AEGIS_OCR_MODEL_DIR` | Pre-bundled EasyOCR weights directory. |
| `AEGIS_STORAGE_PROTOCOL` | `file` (default), `s3`, `gs`, `az`. |

### Side-effect profile (declare these in a manifest)

- **Writes:** the caller-provided output directory only (artifacts + CSV
  reports). No writes to system paths or shell startup files.
- **Network:** on first run, downloads the pinned NER model (~500 MB) and
  EasyOCR weights — unless run offline (`AEGIS_HF_OFFLINE=true`,
  `AEGIS_MODEL_DOWNLOADS=false`) after `scripts/prefetch_models.py`.
- **GPU:** not required. CPU works; `AEGIS_DEVICE=cuda|mps` accelerates.
- **Runtime:** OCR + NER on CPU is minutes-scale per volume and multi-GB
  RSS; measure on your data and set manifest cost bounds accordingly.

## Adapter guides

### Anthropic / Claude Skill

A skill directory whose `SKILL.md` instructs the agent to run
`aegis-deidentify` (preferred) or the MCP tools, then read the envelope. A
worked example — MCP registration plus a skill — is the plugin under
[`plugins/aegis/`](../../plugins/aegis/). Start from
[`SKILL.template.md`](SKILL.template.md); keep the guardrails from
[`DISCLAIMER.md`](DISCLAIMER.md).

### Manifest-based catalog (e.g. NVIDIA MedTech Medical AI Skills)

Wrap one entry point (`aegis-deidentify`) through its documented command.
Point the manifest's output schema at `schemas/envelope.schema.json`, set
`validation.env_pin` from the bounded dependencies in `pyproject.toml`,
declare the side-effect profile above, and add `aegis-verify` as the paired
verifier so the de-identification claim is checked, not just the exit code.
Use `aegis-fixture` to produce the committed synthetic fixture. The scope
and limitations prose in [`DISCLAIMER.md`](DISCLAIMER.md) is written to drop
straight into a skill card's `## Limitations` / `phi_scope_disclaimer`.

### MCP host

Already built: `aegis_mcp_server.py` exposes the pipeline as MCP tools whose
responses are PHI-free by the same envelope contract. Install the `mcp`
extra (`pip install 'monai-aegis[mcp]'`) and register the server; see the
main [README](../../README.md#-mcp-server-ai-agent-interface).
