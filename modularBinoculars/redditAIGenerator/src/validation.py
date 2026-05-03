import hashlib
import re


PROMPT_LEAK_PREFIXES = (
    "sure,",
    "here is",
    "here's",
    "rewritten version:",
    "rewrite:",
    "continuation:",
    "assistant:",
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cleanup_generated_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        first = lines[0].lower()
        if any(first.startswith(prefix) for prefix in PROMPT_LEAK_PREFIXES):
            lines = lines[1:]
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"^```(?:text)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return normalize_spaces(cleaned)


def word_count(text: str) -> int:
    return len(normalize_spaces(text).split())


def length_bin(wc: int) -> str:
    if wc < 100:
        return "short"
    if wc < 300:
        return "medium"
    if wc < 700:
        return "long"
    return "very_long"


def text_fingerprint(text: str) -> str:
    normalized = normalize_spaces(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_generated_text(
    generated_text: str,
    source_text: str,
    min_generated_words: int,
    seen_fingerprints: set[str],
) -> tuple[bool, str]:
    cleaned = cleanup_generated_text(generated_text)
    if not cleaned:
        return False, ""

    if word_count(cleaned) < min_generated_words:
        return False, ""

    if normalize_spaces(cleaned).lower() == normalize_spaces(source_text).lower():
        return False, ""

    fingerprint = text_fingerprint(cleaned)
    if fingerprint in seen_fingerprints:
        return False, ""

    seen_fingerprints.add(fingerprint)
    return True, cleaned
