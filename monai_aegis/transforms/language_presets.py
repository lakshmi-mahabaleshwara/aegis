"""
EasyOCR language presets for multilingual medical imaging.

EasyOCR supports 80+ languages, but languages from different script families
generally cannot be mixed in a single Reader (English is the exception —
it may be paired with any language). Presets group compatible codes so
deployments can enable a script family without hand-picking codes.

Resolution order in :func:`resolve_ocr_languages`:
  1. Explicit ``ocr.languages`` list (if non-empty) wins.
  2. Else ``ocr.language_preset`` expands to a preset list.
  3. Else default ``['en']``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["LANGUAGE_PRESETS", "resolve_ocr_languages"]

# Compatible EasyOCR language groups. English is included in every preset
# because EasyOCR allows pairing ``en`` with any other language, and burnt-in
# PHI on medical devices is frequently English even in non-English locales.
LANGUAGE_PRESETS: Dict[str, List[str]] = {
    "latin": [
        "en", "es", "fr", "de", "it", "pt", "nl", "pl", "sv", "da",
        "no", "cs", "sk", "hu", "ro", "tr", "vi", "id", "ms",
    ],
    "cjk": ["en", "ch_sim", "ch_tra", "ja", "ko"],
    "arabic": ["en", "ar", "fa", "ur", "ug"],
    "cyrillic": ["en", "ru", "uk", "bg", "be", "rs_cyrillic", "mn"],
    "indic": ["en", "hi", "bn", "ta", "te", "kn", "mr", "ne"],
    "medical_eu": ["en", "es", "fr", "de", "it", "pt", "nl", "pl"],
}


def resolve_ocr_languages(ocr_cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Resolve the EasyOCR language list from config.

    Args:
        ocr_cfg: The ``ocr`` section of config.yaml (or ``None``).

    Returns:
        Non-empty list of EasyOCR language codes.

    Raises:
        ValueError: If ``language_preset`` is set to an unknown name.
    """
    ocr_cfg = ocr_cfg or {}
    languages = ocr_cfg.get("languages") or []
    if languages:
        return list(languages)

    preset = (ocr_cfg.get("language_preset") or "").strip().lower()
    if not preset:
        return ["en"]
    if preset not in LANGUAGE_PRESETS:
        known = ", ".join(sorted(LANGUAGE_PRESETS))
        raise ValueError(
            f"Unknown ocr.language_preset {preset!r}. "
            f"Known presets: {known}. Or set ocr.languages explicitly."
        )
    return list(LANGUAGE_PRESETS[preset])
