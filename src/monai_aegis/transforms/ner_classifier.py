"""
MONAI Aegis NER Classifier — PHIClassifier

Semantic PHI detection using Stanford's de-identification NER model,
enhanced with heuristic pre-filters for short OCR fragments.

The classifier operates in two layers:
  1. **Heuristic pre-filter** — catches obvious PHI patterns that the NER model
     may miss on short, context-free OCR fragments (hospital names, date-ID
     combos, institution keywords).
  2. **Stanford NER model** — semantic classification for remaining text using
     the stanford-deidentifier-base transformer model.

All classification rules (PHI labels, clinical allowlist, clinical patterns,
PHI heuristic patterns) are loaded from config.yaml under the ``ner`` section.

Thread-safe: uses threading.local() for per-thread model instances.
"""
import os
import re
import threading
import logging
from typing import List, Dict, Any, Optional

from monai_aegis.config.config_loader import as_bool
from monai_aegis.transforms.utility import AegisIdentityManager

logger = logging.getLogger(__name__)


class PHIClassifier:
    """
    Thread-safe NER-based PHI classifier with configurable heuristic pre-filters.

    All classification rules are loaded from ``config['ner']``:
      - ``phi_labels`` — NER entity types treated as PHI (checked against the
        loaded model's label set at startup)
      - ``age_patterns`` — regex patterns for ages (the model has no AGE label)
      - ``clinical_allowlist`` — known terms that should never be redacted
      - ``clinical_patterns`` — regex patterns for clinical/technical text
      - ``phi_heuristic_patterns`` — regex patterns that indicate PHI

    Classification pipeline:
      0. Check **age patterns** — "57Y", "Age: 57", "F 57Y" → PHI. Runs before
         the clinical layers, whose ultrasound-frequency and probe-ID patterns
         would otherwise claim sex+age overlays like "F 57Y" / "F57Y".
      1. Check **clinical allowlist** — known device/parameter terms → NOT PHI
      2. Check **clinical patterns** — regex for ultrasound parameters → NOT PHI
      3. Check **PHI heuristic patterns** — date-IDs, institution names → PHI
      4. **Stanford NER model** — semantic classification for remaining text

    Thread-safe: uses ``threading.local()`` per-thread model instances.

    Args:
        config: Configuration dict with 'ner' section.

    Example::

        classifier = PHIClassifier(config=config)
        results = classifier.classify_texts(["SKANDA DIAGNOSTIC CENTRE", "D 13.0"])
        # results = [True, False]
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._thread_local = threading.local()
        ner_config = config.get('ner', {})
        self.model_name = ner_config.get('model_name', 'StanfordAIMI/stanford-deidentifier-base')
        # Exact model-repo commit from config.yaml (ner.model_revision) —
        # pinned there so the model cannot silently change under a
        # deployment. None/empty floats on the repo's default branch.
        self.model_revision = ner_config.get('model_revision') or None
        # Air-gapped mode: resolve the model from the local HF cache only —
        # no network calls, even for revision/update checks.
        self.local_files_only = as_bool(ner_config.get('local_files_only'), default=False)
        self.device = ner_config.get('device', 'cpu')

        # Load all classification rules from config
        self.phi_labels = set(ner_config.get('phi_labels', []))
        self.clinical_allowlist = set(
            s.lower() for s in ner_config.get('clinical_allowlist', [])
        )
        self.clinical_patterns = [
            re.compile(p) for p in ner_config.get('clinical_patterns', [])
        ]
        self.phi_heuristic_patterns = [
            re.compile(p) for p in ner_config.get('phi_heuristic_patterns', [])
        ]
        self.age_patterns = [
            re.compile(p) for p in ner_config.get('age_patterns', [])
        ]
        # OCR fragments are the burned-in PHI this pipeline exists to redact,
        # so they are never logged raw. This tokenizer renders them as the
        # same deterministic token the ground-truth report uses, keeping log
        # lines correlatable with the report without disclosing content.
        self._identity = AegisIdentityManager.from_config(config)

    def _safe(self, text: str) -> str:
        """PHI-safe descriptor of an OCR fragment for logging (token + length)."""
        return f"{self._identity.get_token(text)} len={len(text)}"

    @property
    def pipeline(self):
        """Lazily create a per-thread NER pipeline.

        Raises ``ValueError`` (cached, so the model is not reloaded on every
        call) when ``phi_labels`` shares no label with the loaded model —
        callers treat NER errors as PHI, so such a misconfiguration fails
        closed instead of silently preserving every name.
        """
        label_error = getattr(self._thread_local, 'label_error', None)
        if label_error is not None:
            raise label_error
        if not hasattr(self._thread_local, 'pipeline'):
            if self.local_files_only:
                # HF honours these process-wide flags across tokenizer,
                # config, and model loading — the only reliable way to
                # guarantee zero network traffic in air-gapped deployments.
                os.environ.setdefault('HF_HUB_OFFLINE', '1')
                os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
            from transformers import pipeline as hf_pipeline
            logger.info(
                "Loading NER model '%s' (revision: %s, offline: %s) on device '%s' (thread: %s)",
                self.model_name, self.model_revision or 'default',
                self.local_files_only, self.device,
                threading.current_thread().name,
            )
            # Offline resolution is driven by the HF_HUB_OFFLINE /
            # TRANSFORMERS_OFFLINE env vars set above (honoured across the
            # tokenizer, config, and model loads). We deliberately do NOT
            # pass local_files_only through model_kwargs: transformers >= 5
            # forwards its own local_files_only into AutoConfig.from_pretrained,
            # so a duplicate here raises "got multiple values for keyword
            # argument 'local_files_only'" and breaks NER entirely.
            pipe = hf_pipeline(
                'token-classification',
                model=self.model_name,
                revision=self.model_revision,
                device=self.device,
                aggregation_strategy='simple',
            )
            try:
                self.check_phi_labels(pipe)
            except ValueError as exc:
                self._thread_local.label_error = exc
                raise
            self._thread_local.pipeline = pipe
        return self._thread_local.pipeline

    def check_phi_labels(self, pipe: Any) -> None:
        """Compare ``phi_labels`` with the entity labels the loaded model emits.

        A PHI label the model emits but the config omits is silently kept
        visible (e.g. the Stanford model tags doctors ``HCW`` and identifiers
        ``ID``; a config listing ``DOCTOR``/``IDNUM`` never redacts them), so
        mismatches are surfaced here rather than discovered in the output:

        - no overlap at all → ``ValueError`` (NER could never flag anything);
        - configured labels the model never emits → warning (dead config);
        - model labels not configured → info (deliberately non-PHI, e.g. VENDOR).

        Models that expose no ``config.id2label`` mapping are not checked.
        """
        id2label = getattr(getattr(getattr(pipe, 'model', None), 'config', None), 'id2label', None)
        if not isinstance(id2label, dict):
            return
        model_labels = {re.sub(r'^[BI]-', '', str(label)) for label in id2label.values()} - {'O'}
        if not model_labels:
            return
        if not (self.phi_labels & model_labels):
            message = (
                f"ner.phi_labels {sorted(self.phi_labels)} shares no label with NER model "
                f"'{self.model_name}', which emits {sorted(model_labels)} — NER could never "
                f"flag PHI. Update ner.phi_labels to this model's labels."
            )
            logger.error(message)
            raise ValueError(message)
        unknown = sorted(self.phi_labels - model_labels)
        if unknown:
            logger.warning(
                "ner.phi_labels lists labels the NER model never emits (no effect): %s. "
                "Model labels: %s", unknown, sorted(model_labels),
            )
        not_phi = sorted(model_labels - self.phi_labels)
        if not_phi:
            logger.info("NER labels treated as non-PHI (not in ner.phi_labels): %s", not_phi)

    def _is_age(self, text: str) -> bool:
        """Check if text matches an age pattern (redacted ahead of clinical rules)."""
        return any(p.search(text) for p in self.age_patterns)

    def _is_clinical(self, text: str) -> bool:
        """Check if text matches known clinical/technical terms."""
        normalized = text.strip().lower()
        if normalized in self.clinical_allowlist:
            return True
        return any(p.search(text) for p in self.clinical_patterns)

    def _is_phi_heuristic(self, text: str) -> bool:
        """Check if text matches heuristic PHI patterns."""
        return any(p.search(text) for p in self.phi_heuristic_patterns)

    def classify_line(self, fragments: List[str]) -> List[bool]:
        """Classify the OCR fragments of one text line *in context*.

        OCR splits a line into word groups, and a fragment judged alone
        loses the context that makes it PHI: in "Robinson, Brandy S." the
        surname is caught but "Brandy S." alone is not; in
        "08/01/2021 23:16" the date is caught but "23:16" alone is not.

        The fragments are joined back into their line (single spaces) and
        PHI *spans* are located in it — age patterns, PHI heuristic
        patterns, and PHI-labelled NER entities with their character
        offsets. A fragment is flagged when it overlaps a span, so a name
        on the line pulls in only the fragments it covers, not the whole
        line. A fragment the clinical layer preserves on its own is never
        flagged by context. On an NER error every other fragment is
        flagged (fail closed, as in :meth:`classify_texts`).

        Callers OR the result with the per-fragment flags from
        :meth:`classify_texts`: line context only ever adds redaction.

        Args:
            fragments: The fragments of one line, left to right.

        Returns:
            One flag per fragment — True if line context marks it as PHI.
        """
        parts = [f.strip() for f in fragments]
        offsets, pos = [], 0
        for part in parts:
            offsets.append((pos, pos + len(part)))
            pos += len(part) + 1
        line = " ".join(parts)
        protected = [not part or self._is_clinical(part) for part in parts]
        if not line.strip() or all(protected):
            return [False] * len(parts)

        spans = []
        for pattern in self.age_patterns + self.phi_heuristic_patterns:
            spans.extend(m.span() for m in pattern.finditer(line))

        # Same short-text context wrapping as classify_texts.
        prefix = "The report mentions " if len(line.split()) <= 3 else ""
        try:
            for ent in self.pipeline(f"{prefix}{line}." if prefix else line):
                label = ent['entity_group'].replace('B-', '').replace('I-', '')
                if label not in self.phi_labels:
                    continue
                start, end = ent.get('start'), ent.get('end')
                if start is None or end is None:
                    spans.append((0, len(line)))       # no offsets: whole line
                else:
                    spans.append((start - len(prefix), end - len(prefix)))
        except Exception as e:
            logger.error("NER line classification error for %s: %s",
                         self._safe(line), type(e).__name__)
            spans.append((0, len(line)))

        return [
            not prot and any(s < b and a < e for s, e in spans)
            for (a, b), prot in zip(offsets, protected)
        ]

    def classify_texts(self, texts: List[str]) -> List[bool]:
        """
        Classify a list of text strings as PHI or non-PHI.

        Classification order:
          1. Empty/whitespace → NOT PHI
          2. Age patterns → PHI (before the clinical rules — see class docs)
          3. Clinical allowlist/patterns → NOT PHI
          4. PHI heuristic patterns → PHI
          5. NER model → PHI if any PHI entity detected

        Args:
            texts: List of OCR-detected text strings.

        Returns:
            List of booleans — True if the text contains PHI, False otherwise.
        """
        results = []
        texts_for_ner = []  # (index, text) pairs needing NER
        
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                results.append(False)
                continue

            # Layer 0: Ages → redact. Must precede the clinical layer: "F 57Y"
            # would otherwise match the ultrasound-frequency pattern.
            if self._is_age(text):
                logger.debug("Age (redacted): %s", self._safe(text))
                results.append(True)
                continue

            # Layer 1: Clinical allowlist/patterns → preserve
            if self._is_clinical(text):
                logger.debug("Clinical term (preserved): %s", self._safe(text))
                results.append(False)
                continue

            # Layer 2: PHI heuristic patterns → redact
            if self._is_phi_heuristic(text):
                logger.debug("Heuristic PHI (redacted): %s", self._safe(text))
                results.append(True)
                continue

            # Layer 3: Defer to NER model
            results.append(None)  # placeholder
            texts_for_ner.append((i, text))

        # Batch NER classification for remaining texts
        if texts_for_ner:
            ner_texts = [t for _, t in texts_for_ner]
            # Wrap short fragments in context to help the NER model
            contextualized = [
                f"The report mentions {t}." if len(t.split()) <= 3 else t
                for t in ner_texts
            ]
            
            for (idx, orig_text), ctx_text in zip(texts_for_ner, contextualized):
                try:
                    entities = self.pipeline(ctx_text)
                    is_phi = any(
                        ent['entity_group'].replace('B-', '').replace('I-', '') in self.phi_labels
                        for ent in entities
                    )
                    if is_phi:
                        # entity_group values are category labels (PATIENT,
                        # DATE, ...), never the matched text — safe to log.
                        logger.debug("NER classified as PHI: %s → %s", self._safe(orig_text),
                                     [e['entity_group'] for e in entities])
                    else:
                        logger.debug("NER classified as non-PHI (preserved): %s", self._safe(orig_text))
                    results[idx] = is_phi
                except Exception as e:
                    logger.error("NER classification error for %s: %s", self._safe(orig_text),
                                 type(e).__name__)
                    results[idx] = True  # default to PHI for safety

        return results
