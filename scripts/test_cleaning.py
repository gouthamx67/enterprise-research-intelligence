from src.rag_engine.preprocessing.cleaning import clean_text

from src.rag_engine.preprocessing.cleaning import (
    clean_text,
    looks_like_page_number,
    should_remove_block,
)

examples = [
    "   The company   increased revenue.\n\n\n",
    "Revenue increased\n18% in Q3.",
    "Hello\r\nWorld",
    "1",
    "18%",
    "Item 1A",
    "2026",
]


for example in examples:

    cleaned = clean_text(example)

    print("BEFORE:", repr(example))
    print("AFTER: ", repr(cleaned))
    print("PAGE NUMBER:", looks_like_page_number(cleaned))
    print("REMOVE:", should_remove_block(cleaned))

    print("-" * 50)