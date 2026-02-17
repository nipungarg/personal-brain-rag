import re
from .load import load_text


def clean_text(filename: str) -> str:
    text = load_text(filename)

    # Normalize whitespace: tabs and multiple spaces -> single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines to at most one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace and newlines
    text = text.strip()

    # Fix double punctuation (e.g.. -> e.g.)
    text = re.sub(r"e\.g\.\.", "e.g.", text)

    # Remove duplicate consecutive words (e.g. "the the" -> "the")
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)

    # Remove common footer/metadata lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^---\s*\[.*\]\s*---\s*$", stripped):
            continue
        if stripped.startswith("TODO:"):
            continue
        if re.match(r"^Version:.*Last updated:", stripped):
            continue
        if stripped == "[REVIEW NEEDED]":
            continue
        if re.match(r"^\*{2,}\s*.*\s*\*{2,}\s*$", stripped):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()

    return text