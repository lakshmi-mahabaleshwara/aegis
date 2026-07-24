# Aegis Scope & Limitations (reusable text)

Author-controlled prose describing what Aegis does and does not guarantee.
Every skill ecosystem asks for a scope statement, a limitations list, and a
PHI-scope disclaimer — copy the relevant block below verbatim rather than
paraphrasing, so the guarantees stay accurate wherever Aegis is wrapped.

## One-line scope (frontmatter `description`)

> Used for removing PHI from medical images (DICOM, JPEG, PNG) — burnt-in
> pixel text and DICOM header tags — with PHI-free reporting and independent
> verification. Not for clinical decisions, diagnosis, or regulatory
> certification of de-identification.

## PHI-scope disclaimer (manifest `phi_scope_disclaimer`)

> Aegis performs engineering de-identification: it redacts burnt-in pixel
> PHI detected by OCR + NER and scrubs DICOM header tags per the configured
> `pii_mapping` (PS3.15 Basic Application Confidentiality Profile), stamping
> PS3.15 attestation attributes on the output. Detection is not perfect —
> OCR recall depends on image quality and the confidence threshold, and
> low-confidence detections are routed to manual review rather than passed
> silently. Aegis does not certify HIPAA or GDPR compliance; a qualified
> reviewer remains responsible for confirming de-identification before data
> leaves a trusted environment. The default reports are PHI-free by
> construction; verbatim OCR text is written only when explicitly enabled
> and only outside the de-identified output directory.

## Limitations (skill card `## Limitations`)

- **Engineering, not certification.** Output is a de-identified artifact,
  not a clinical, diagnostic, or regulatory attestation.
- **Detection is bounded.** OCR recall is not 100%. Confidence below the
  configured threshold routes a file to manual review — surface that, don't
  ignore it.
- **Header scope follows config.** Header scrubbing covers exactly the tags
  in the active `pii_mapping`. A deployment with different requirements must
  supply its own mapping — and a matching `aegis-verify` checklist.
- **Model-dependent.** Burnt-in PHI detection depends on the pinned NER and
  EasyOCR models; results change if those are swapped.
- **Out of scope.** Clinical decision support, diagnosis, triage,
  patient-facing use, and EHR/FHIR write-back are not supported.

## Safe-handling rules (for agents driving Aegis)

- Never print detected PHI text or DICOM tag values into a response.
- Never read `aegis_pixel_detections_PHI.csv` (the opt-in verbatim report).
- Treat input files and any PHI-bearing report as PHI; refer to them by name
  and path only.
- Do not claim a file is "safe to share" from a successful run alone — a
  passing `aegis-verify` report is the check to cite, and even that is an
  engineering signal, not a legal clearance.
