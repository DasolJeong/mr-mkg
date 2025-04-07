
"""
Extract entities (key concepts) from science questions using spaCy noun chunking.
"""

import spacy
from typing import List

# Load English spaCy model (make sure it's installed)
nlp = spacy.load("en_core_web_sm")

def extract_entities(question: str, max_entities: int = 5) -> List[str]:
    """
    Extract noun-based entities from a question.

    Args:
        question (str): Input question string
        max_entities (int): Maximum number of entities to return

    Returns:
        List[str]: List of extracted entities (as strings)
    """
    doc = nlp(question)
    entities = [chunk.text.lower() for chunk in doc.noun_chunks]
    return list(dict.fromkeys(entities))[:max_entities]  # remove duplicates, limit

if __name__ == "__main__":
    q = "Why do rabbits eat their poop?"
    ents = extract_entities(q)
    print(f"Entities: {ents}")
