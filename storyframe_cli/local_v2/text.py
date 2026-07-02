from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

try:  # pragma: no cover - exercised when optional dependency is installed.
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback keeps source tests lightweight.
    fuzz = None


IGNORE_WORDS = {
    "vooks",
    "vo0ks",
    "vook",
    "vooksy",
    "wooks",
    "storybooks",
    "storybook",
    "tm",
}

REJECT_PHRASES = {
    "app store",
    "app for free",
    "ask your grown up",
    "brought to life",
    "continue watching",
    "for more stories",
    "google play",
    "new releases",
    "popular titles",
    "subscribe",
    "thanks for watching",
    "thank you for watching",
    "try the app",
    "created by",
    "written by",
    "illustrated by",
    "copyright",
    "all rights reserved",
    "age elementary",
    "available from",
    "based on the storybook",
    "cover design",
    "designed by",
    "edited by",
    "popular titles",
    "published in",
    "published by",
    "start exploring more fun stories",
    "watch now",
    "you'll be glad you did",
}


def clean_text(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    tokens: list[str] = []
    for token in text.split():
        token = token.strip("'")
        if not token:
            continue
        if token in IGNORE_WORDS:
            continue
        if re.search(r"[a-z]", token) and re.search(r"\d", token):
            continue
        if token in FUSED_TOKEN_SPLITS:
            tokens.extend(FUSED_TOKEN_SPLITS[token])
            continue
        if len(token) == 1 and token not in {"a", "i"}:
            continue
        tokens.append(token)
    return " ".join(tokens)


FUSED_TOKEN_SPLITS = {
    "bearpoo": ["bear", "poo"],
    "doesa": ["does", "a"],
    "eightall": ["eight", "all"],
    "firefliesgotcha": ["fireflies", "gotcha"],
    "ihear": ["i", "hear"],
    "inthe": ["in", "the"],
    "it'sa": ["it's", "a"],
    "moonsnail": ["moon", "snail"],
    "musselshell": ["mussel", "shell"],
    "seashe": ["seashell"],
    "seashel": ["seashell"],
    "seashelt": ["seashell"],
    "tenboth": ["ten", "both"],
    "whelkshell": ["whelk", "shell"],
}


def has_reject_phrase(text: str) -> bool:
    padded = f" {clean_text(text)} "
    return any(phrase in padded for phrase in REJECT_PHRASES)


def token_set(text: str) -> set[str]:
    return set(clean_text(text).split())


def token_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if fuzz is not None:
        return float(fuzz.ratio(left, right)) / 100.0
    return SequenceMatcher(None, left, right).ratio()


def tokens_match(left: str, right: str) -> bool:
    if left == right:
        return True
    score = token_similarity(left, right)
    min_len = min(len(left), len(right))
    max_len = max(len(left), len(right))
    if min_len <= 2:
        return score >= 0.92
    if min_len == 3:
        one_char_prefix_missing = max_len == min_len + 1 and (
            left.endswith(right) or right.endswith(left)
        )
        return one_char_prefix_missing or score >= 0.88
    if min_len == 4:
        return score >= 0.84
    return score >= 0.78


def similarity(left: str, right: str) -> float:
    left_clean = clean_text(left)
    right_clean = clean_text(right)
    if not left_clean or not right_clean:
        return 0.0
    if left_clean == right_clean:
        return 1.0
    left_tokens = set(left_clean.split())
    right_tokens = set(right_clean.split())
    overlap = len(left_tokens & right_tokens)
    containment = overlap / max(1, min(len(left_tokens), len(right_tokens)))
    jaccard = overlap / max(1, len(left_tokens | right_tokens))
    sequence = SequenceMatcher(None, left_clean, right_clean).ratio()
    return max(containment * 0.70 + jaccard * 0.30, sequence)


def target_coverage(target: str, observed: str) -> float:
    target_tokens = token_set(target)
    if not target_tokens:
        return 0.0
    observed_tokens = token_set(observed)
    return fuzzy_overlap_count(target_tokens, observed_tokens) / len(target_tokens)


def extra_token_ratio(target: str, observed: str) -> float:
    observed_tokens = token_set(observed)
    if not observed_tokens:
        return 0.0
    target_tokens = token_set(target)
    matched = fuzzy_overlap_count(observed_tokens, target_tokens)
    return (len(observed_tokens) - matched) / len(observed_tokens)


def fuzzy_overlap_count(left_tokens: set[str], right_tokens: set[str]) -> int:
    remaining = list(right_tokens)
    matched = 0
    for token in left_tokens:
        best_index = best_token_match_index(token, remaining)
        if best_index is None:
            continue
        matched += 1
        remaining.pop(best_index)
    return matched


def best_token_match_index(token: str, candidates: list[str]) -> int | None:
    best_index: int | None = None
    best_score = 0.0
    for index, candidate in enumerate(candidates):
        if not tokens_match(token, candidate):
            continue
        score = token_similarity(token, candidate)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def corrected_text_with_reference(
    text: str,
    reference: str,
    *,
    allow_insertions: bool = False,
    keep_unmatched: bool = True,
    max_inserted_tokens: int = 4,
) -> str:
    tokens = clean_text(text).split()
    reference_tokens = clean_text(reference).split()
    if len(tokens) < 2 or not reference_tokens:
        return text

    corrected: list[str] = []
    last_reference_index = -1
    changed = False
    for token in tokens:
        match = best_ordered_reference_match(token, reference_tokens, last_reference_index + 1)
        if match is None:
            if keep_unmatched or should_keep_unmatched_ocr_token(token):
                corrected.append(token)
            else:
                changed = True
            continue
        index, reference_token = match
        if allow_insertions and index > last_reference_index + 1 >= 1:
            missing_tokens = reference_tokens[last_reference_index + 1:index]
            if len(missing_tokens) <= max_inserted_tokens:
                corrected.extend(missing_tokens)
                changed = True
        last_reference_index = index
        replacement = (
            reference_token
            if should_replace_with_reference_token(token, reference_token)
            else token
        )
        corrected.append(replacement)
        changed = changed or replacement != token
    if not changed:
        return text
    return " ".join(corrected)


def should_replace_with_reference_token(token: str, reference_token: str) -> bool:
    if token == reference_token:
        return False
    if is_simple_plural_variant(token, reference_token):
        return False
    score = token_similarity(token, reference_token)
    min_len = min(len(token), len(reference_token))
    if min_len <= 3:
        return tokens_match(token, reference_token)
    if min_len <= 5:
        return score >= 0.78
    return len(token) == len(reference_token) and score >= 0.84


def should_keep_unmatched_ocr_token(token: str) -> bool:
    vowel_count = sum(1 for char in token if char in "aeiou")
    consonant_count = sum(
        1 for char in token if "a" <= char <= "z" and char not in "aeiou"
    )
    if len(token) >= 5 and consonant_count <= 1 and vowel_count >= len(token) - 1:
        return False
    if len(token) > 3:
        return True
    common_short_words = {
        "a",
        "am",
        "an",
        "as",
        "at",
        "be",
        "by",
        "do",
        "go",
        "he",
        "hi",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "no",
        "of",
        "on",
        "or",
        "poo",
        "she",
        "the",
        "to",
        "up",
        "we",
        "who",
        "you",
    }
    return token in common_short_words


def is_simple_plural_variant(left: str, right: str) -> bool:
    return (left.endswith("s") and left[:-1] == right) or (
        right.endswith("s") and right[:-1] == left
    )


def best_ordered_reference_match(
    token: str,
    reference_tokens: list[str],
    start_index: int,
) -> tuple[int, str] | None:
    search_ranges = (range(start_index, len(reference_tokens)), range(0, len(reference_tokens)))
    best: tuple[float, int, str] | None = None
    for search_range in search_ranges:
        for index in search_range:
            candidate = reference_tokens[index]
            if not tokens_match(token, candidate):
                continue
            score = token_similarity(token, candidate)
            if best is None or score > best[0]:
                best = (score, index, candidate)
        if best is not None:
            break
    if best is None:
        return None
    return best[1], best[2]
