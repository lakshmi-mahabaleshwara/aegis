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
import re
import threading
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PHIClassifier:
    """
    Thread-safe NER-based PHI classifier with configurable heuristic pre-filters.

    All classification rules are loaded from ``config['ner']``:
      - ``phi_labels`` — NER entity types treated as PHI
      - ``clinical_allowlist`` — known terms that should never be redacted
      - ``clinical_patterns`` — regex patterns for clinical/technical text
      - ``phi_heuristic_patterns`` — regex patterns that indicate PHI

    Classification pipeline:
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

    @property
    def pipeline(self):
        """Lazily create a per-thread NER pipeline."""
        if not hasattr(self._thread_local, 'pipeline'):
            from transformers import pipeline as hf_pipeline
            logger.info(
                "Loading NER model '%s' on device '%s' (thread: %s)",
                self.model_name, self.device, threading.current_thread().name
            )
            self._thread_local.pipeline = hf_pipeline(
                'token-classification',
                model=self.model_name,
                device=self.device,
                aggregation_strategy='simple',
            )
        return self._thread_local.pipeline

    def _is_clinical(self, text: str) -> bool:
        """Check if text matches known clinical/technical terms."""
        normalized = text.strip().lower()
        if normalized in self.clinical_allowlist:
            return True
        return any(p.search(text) for p in self.clinical_patterns)

    def _is_phi_heuristic(self, text: str) -> bool:
        """Check if text matches heuristic PHI patterns."""
        return any(p.search(text) for p in self.phi_heuristic_patterns)

    def classify_texts(self, texts: List[str]) -> List[bool]:
        """
        Classify a list of text strings as PHI or non-PHI.

        Classification order:
          1. Empty/whitespace → NOT PHI
          2. Clinical allowlist/patterns → NOT PHI
          3. PHI heuristic patterns → PHI
          4. NER model → PHI if any PHI entity detected

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

            # Layer 1: Clinical allowlist/patterns → preserve
            if self._is_clinical(text):
                logger.debug("Clinical term (preserved): '%s'", text)
                results.append(False)
                continue

            # Layer 2: PHI heuristic patterns → redact
            if self._is_phi_heuristic(text):
                logger.debug("Heuristic PHI (redacted): '%s'", text)
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
                        logger.debug("NER classified as PHI: '%s' → %s", orig_text,
                                     [e['entity_group'] for e in entities])
                    else:
                        logger.debug("NER classified as non-PHI (preserved): '%s'", orig_text)
                    results[idx] = is_phi
                except Exception as e:
                    logger.error("NER classification error for '%s': %s", orig_text, e)
                    results[idx] = True  # default to PHI for safety

        return results
