# reward_custom.py
import re
from typing import Any, Dict, Optional, Union

# ---------- extraction rules ----------

def _get_last_nonempty_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    for ln in reversed(lines):
        if ln:  # non-empty
            return ln
    return ""

def _extract_answer_last_line(text: str) -> Optional[str]:
    """
    Rule: Only look for 'Answer: XXX' in the LAST non-empty line.
    No fallback.
    """
    last_line = _get_last_nonempty_line(text)
    if not last_line:
        return None
    m = re.search(r"Answer:\s*(.+)\s*$", last_line, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def _extract_boxed_anywhere(text: str) -> Optional[str]:
    """
    Rule: Search \\boxed{...} anywhere in full text; take the last one.
    No fallback.
    """
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None

def _extract_by_source(text: str, data_source: str) -> Optional[str]:
    """
    Source-specific extraction policy (per your rule #4).
    """
    if data_source in ("math-dapo", "mcqa"):
        return _extract_answer_last_line(text)
    if data_source == "openscience":
        return _extract_boxed_anywhere(text)

    # default for other sources (safe best-effort):
    # try boxed first, then last-line Answer
    return _extract_boxed_anywhere(text) or _extract_answer_last_line(text)

# ---------- normalization / comparison ----------

def _strip_latex_and_commas(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    # remove common latex wrappers
    s = re.sub(r"(\$|\\\(|\\\)|\\\[|\\\])", "", s)
    # unwrap \text{...} if present
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return s.strip()

def _normalize(s: str) -> str:
    return _strip_latex_and_commas(s).lower()

def _parse_full_int(s: str) -> Optional[str]:
    """
    If s is a *pure* integer string (after strip), return canonical int string.
    Else None.
    """
    s2 = _strip_latex_and_commas(s)
    if re.fullmatch(r"-?\d+", s2):
        # canonicalize e.g., "+01" -> "1"
        return str(int(s2))
    return None

def _compare_pred_gt(pred: str, gt: str) -> bool:
    """
    - If both are pure integers: compare numerically.
    - Else: compare normalized strings exactly.
    """
    p_int = _parse_full_int(pred)
    g_int = _parse_full_int(gt)
    if p_int is not None and g_int is not None:
        return p_int == g_int
    return _normalize(pred) == _normalize(gt)

# ---------- compute reward ----------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    verl RewardManager will call this.
    """
    # compat: ground_truth might be a dict like {"ground_truth": "...", "style": "..."}
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("ground_truth", "")

    pred = _extract_by_source(solution_str or "", data_source or "")
    gt   = _extract_by_source(str(ground_truth or ""), data_source or "")

    if pred is None or gt is None:
        return 0.0

    return 1.0 if _compare_pred_gt(pred, gt) else 0.0
