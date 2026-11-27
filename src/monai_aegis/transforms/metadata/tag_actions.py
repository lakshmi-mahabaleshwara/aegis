"""
Configured header scrub: ``pii_mapping`` parsing and the
REMOVE / ZERO / DUMMY / KEEP actions.

Two halves:

* **Configuration** (module functions) — turn the ``pii_mapping`` section of
  ``config.yaml`` into a ``{Tag: ACTION}`` table once, at construction time,
  failing loudly on a mistyped tag or unknown action.
* **Application** (:class:`TagActionScrubber`) — walk a dataset tree applying
  one action per configured tag, then strip private tags.

Walk order (observable through the audit trail): top-level elements are
visited in tag order; when a sequence (``SQ``) element is reached its items
are scrubbed *first* (depth-first), then the sequence element's own action
— if one is configured — is applied. So a REMOVE on a sequence tag still
audits every nested action before the sequence disappears. Private-tag
removal runs as a separate pass after the actions and is not audited.
"""
from typing import Any, Dict, Tuple

import pydicom
import pydicom.tag

from monai_aegis.transforms.exceptions import MetadataScrubError
from monai_aegis.transforms.metadata.audit import TagAuditTrail, keyword_for
from monai_aegis.transforms.metadata.constants import (
    ACTION_DUMMY,
    ACTION_KEEP,
    ACTION_REMOVE,
    ACTION_ZERO,
    ARRAY_TRANSFORM_NAME,
    AUTO_BASE,
    BINARY_VRS,
    HEX_BASE,
    HEX_PREFIX,
    TAG_KEY_BRACKETS,
    TAG_KEY_SEPARATOR,
    VALID_ACTIONS,
    VR_MAX_LEN,
    VR_SEQUENCE,
    ZERO_BINARY_VALUE,
    ZERO_TEXT_VALUE,
)
from monai_aegis.transforms.utility import AegisIdentityManager

__all__ = [
    "PiiActions",
    "TagActionScrubber",
    "build_pii_actions",
    "fit_token_to_vr",
    "parse_hex_part",
    "parse_tag_key",
]

# Parsed pii_mapping: tag -> upper-cased action (one of VALID_ACTIONS).
PiiActions = Dict[pydicom.tag.BaseTag, str]


# ---------------------------------------------------------------------------
# Configuration: pii_mapping -> {Tag: ACTION}
# ---------------------------------------------------------------------------

def parse_hex_part(s: str) -> int:
    """Parse one DICOM tag group/element string as a hex integer.

    Accepts both plain hex (``0010``) and explicit ``0x``-prefixed hex
    (``0x0010``).  Plain digits are always interpreted as base-16 via
    ``int(s, 16)`` — never as decimal or octal — so leading zeros are
    harmless and no ``SyntaxError`` can arise from the string value.
    """
    s = s.strip()
    if s.lower().startswith(HEX_PREFIX):
        return int(s, AUTO_BASE)   # 0x-prefix: let Python auto-detect (always 16)
    return int(s, HEX_BASE)        # plain hex digits: explicit base-16


def parse_tag_key(tag_str: str) -> Tuple[int, int]:
    """Parse a DICOM tag string key into a ``(group, element)`` int tuple.

    Accepts any of these formats from ``pii_mapping`` in ``config.yaml``:

    * ``'(0010,0010)'``   — preferred, standard DICOM notation (no space)
    * ``'(0010, 0010)'``  — with space after comma (legacy, still accepted)
    * ``'(0x0010,0x0010)'`` — explicit hex prefix (also accepted)

    Args:
        tag_str: DICOM tag as a string key from ``pii_mapping``.

    Returns:
        ``(group, element)`` as a tuple of ints.

    Raises:
        ValueError: If ``tag_str`` does not contain exactly two
            comma-separated parts.
    """
    inner = tag_str.strip().strip(TAG_KEY_BRACKETS)
    parts = inner.split(TAG_KEY_SEPARATOR)
    if len(parts) != 2:
        raise ValueError(f"Invalid DICOM tag format: {tag_str!r}")
    return (parse_hex_part(parts[0]), parse_hex_part(parts[1]))


def build_pii_actions(pii_mapping: Dict[str, Any]) -> PiiActions:
    """Parse configured tag actions once so recursive scrubbing can reuse them.

    Raises:
        ValueError: If a tag key cannot be parsed or an action is not
            one of REMOVE / ZERO / DUMMY / KEEP. Misconfiguration fails the
            pipeline at construction rather than silently skipping the
            rule (which would pass PHI through unscrubbed).
    """
    actions: PiiActions = {}
    for tag_str, action in pii_mapping.items():
        try:
            group, element = parse_tag_key(tag_str)
            tag = pydicom.tag.Tag(group, element)
        except Exception as e:
            raise ValueError(
                f"Invalid DICOM tag {tag_str!r} in pii_mapping: {e}"
            ) from e
        action_upper = str(action).upper()
        if action_upper not in VALID_ACTIONS:
            raise ValueError(
                f"Unknown pii_mapping action {action!r} for tag {tag_str} — "
                f"valid actions: {sorted(VALID_ACTIONS)}. "
                "Refusing to run: a mistyped action would leave this tag's "
                "PHI in place while the audit reported it as redacted."
            )
        actions[tag] = action_upper
    return actions


def fit_token_to_vr(token: str, vr: str) -> str:
    """Truncate a DUMMY token so it conforms to the element's VR length limit."""
    limit = VR_MAX_LEN.get(vr)
    if limit is not None and len(token) > limit:
        return token[:limit]
    return token


# ---------------------------------------------------------------------------
# Application: walk a dataset tree
# ---------------------------------------------------------------------------

class TagActionScrubber:
    """Applies a parsed ``pii_mapping`` to datasets and nested sequence items.

    Holds only immutable-after-construction state (the action table and the
    shared identity manager), so one instance is safe to share across
    threads; the audit trail is always passed in by the caller.

    Args:
        pii_actions: Output of :func:`build_pii_actions`.
        identity_manager: Token source for DUMMY values.
    """

    def __init__(self, pii_actions: PiiActions, identity_manager: AegisIdentityManager) -> None:
        self._pii_actions = pii_actions
        self._identity_manager = identity_manager

    @property
    def pii_actions(self) -> PiiActions:
        """The parsed ``{Tag: ACTION}`` table this scrubber applies."""
        return self._pii_actions

    def apply_action(
        self,
        ds: pydicom.Dataset,
        tag: pydicom.tag.BaseTag,
        action: str,
        trail: TagAuditTrail,
    ) -> None:
        """Apply one configured action to a single element on a dataset.

        Records the audit row on *trail* (owned by the current scrub call)
        rather than instance state, so concurrent calls cannot interleave
        their audit trails. A tag absent from *ds* is a no-op.
        """
        if tag not in ds:
            return
        keyword = keyword_for(tag)
        if action == ACTION_REMOVE:
            del ds[tag]
        elif action == ACTION_ZERO:
            vr = ds[tag].VR
            ds[tag].value = ZERO_BINARY_VALUE if vr in BINARY_VRS else ZERO_TEXT_VALUE
        elif action == ACTION_DUMMY:
            token = self._identity_manager.get_token(str(ds[tag].value))
            ds[tag].value = fit_token_to_vr(token, ds[tag].VR)
        elif action == ACTION_KEEP:
            pass  # deliberate retention — value untouched, recorded below
        else:
            # Unreachable when actions come from build_pii_actions (which
            # validates), but guards any future caller: an unknown action
            # must never fall through and record a redaction that did not
            # happen.
            raise MetadataScrubError(
                f"Unhandled pii_mapping action {action!r} for tag {tag}",
                transform=ARRAY_TRANSFORM_NAME,
            )
        trail.record_action(tag, keyword, action)

    def scrub_recursive(self, ds: pydicom.Dataset, trail: TagAuditTrail) -> None:
        """Apply configured actions to this dataset and every nested sequence item."""
        for elem in list(ds):
            if elem.VR == VR_SEQUENCE:
                for item in elem.value:
                    if isinstance(item, pydicom.Dataset):
                        self.scrub_recursive(item, trail)
            action = self._pii_actions.get(elem.tag)
            if action is not None:
                self.apply_action(ds, elem.tag, action, trail)

    @staticmethod
    def remove_private_tags_recursive(ds: pydicom.Dataset) -> None:
        """Remove private tags from this dataset and nested sequence items."""
        ds.remove_private_tags()
        for elem in ds:
            if elem.VR == VR_SEQUENCE:
                for item in elem.value:
                    if isinstance(item, pydicom.Dataset):
                        TagActionScrubber.remove_private_tags_recursive(item)
