"""Turn sequences of gesture labels into short, natural English sentences."""
from __future__ import annotations

import re
from typing import List

_GESTURE_PHRASE = {
    "HELLO": "hello",
    "THANK_YOU": "thank you",
    "THANK YOU": "thank you",
    "YES": "yes",
    "NO": "no",
    "PLEASE": "please",
    "SORRY": "sorry",
    "HELP": "help",
    "WATER": "water",
    "FOOD": "food",
    "WANT": "want",
    "I": "I",
    "YOU": "you",
    "GOOD": "good",
    "BAD": "bad",
    "SOS": "emergency",
    "DANGER": "danger",
    "TOILET": "the restroom",
    "UNKNOWN": "",
}


def _normalize_token(raw: str) -> str:
    t = (raw or "").strip().upper().replace(" ", "_")
    return t


def gesture_tokens_to_sentence(tokens: List[str]) -> str:
    """
    Convert recent gesture tokens (e.g. ['WATER','WANT']) into a short, grammatically correct sentence.
    """
    if not tokens:
        return ""
    
    seen = []
    # Deduplicate sequential tokens
    for t in tokens:
        n = _normalize_token(t)
        if not n or n == "UNKNOWN":
            continue
        if not seen or seen[-1] != n:
            seen.append(n)
            
    if not seen:
        return ""
        
    words = []
    for g in seen:
        phrase = _GESTURE_PHRASE.get(g) or _GESTURE_PHRASE.get(g.replace("_", " "))
        if phrase:
            words.append(phrase.lower())
        else:
            words.append(g.replace("_", " ").lower())
            
    if not words:
        return " ".join(seen).capitalize() + "."

    # Contextual grammar replacement engine
    joined = " ".join(words)
    
    # Define grammatical context replacements (Noun + Verb to Subject + Verb + Object)
    replacements = [
        (r"\bwater want\b", "I want water"),
        (r"\bwant water\b", "I want water"),
        (r"\bfood want\b", "I want food"),
        (r"\bwant food\b", "I want food"),
        (r"\bi food want\b", "I want food"),
        (r"\bi water want\b", "I want water"),
        (r"\bhelp please\b", "Please help me"),
        (r"\bplease help\b", "Please help me"),
        (r"\btoilet want\b", "I need to use the restroom"),
    ]
    
    for pattern, replacement in replacements:
        if re.search(pattern, joined):
            joined = re.sub(pattern, replacement, joined)
            break # Break after first major contextual structure match
    
    # Catch-all simple patterns
    if joined.strip() == "help":
        joined = "I need help"
    elif joined.strip() == "water":
        joined = "Water"
    elif joined.strip() == "food":
        joined = "Food"
    
    # Capitalize and append punctuation
    sentence = joined.strip()
    sentence = re.sub(r"\s+", " ", sentence)
    if not sentence:
        return ""
        
    return sentence[0].upper() + sentence[1:] + ("" if sentence.endswith(".") else ".")

