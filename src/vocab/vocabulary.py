import os
import math
from collections import Counter
from typing import Dict, Optional

from pydantic import BaseModel
from src.models.chunks import CollectionEnum
from src.pos.pos_tagger import pos_tagger


STOPWORDS = {
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "let", "make",
    "get", "got",
    "go", "gone",
    "see", "seen",
    "say", "said",
    "take", "took",
    "come", "came",
    "put",
    "form",
    "make",
    "use",
    "using",
    "used",
    "based",
}


class WordInfo(BaseModel):
    document_frequency: Optional[int]
    idf: Optional[float]
    collection_vote: CollectionEnum


def filter_tokens(text: str) -> list[str]:
    tagged_tokens = pos_tagger(text)
    return [
        token.lower()
        for token, pos in tagged_tokens
        if pos.name in ("NOUN", "VERB") and token.lower() not in STOPWORDS
    ]


ROOT_DATA_DIR = "data/extracted"

COLLECTION_DIRS = {
    CollectionEnum.ACADEMICS: os.path.join(ROOT_DATA_DIR, "academics"),
    CollectionEnum.FACULTY: os.path.join(ROOT_DATA_DIR, "faculty"),
    CollectionEnum.STUDENTS: os.path.join(ROOT_DATA_DIR, "students"),
}


def compute_collection_statistics() -> Dict[str, WordInfo]:
    collection_frequency = {collection: Counter() for collection in COLLECTION_DIRS}
    document_frequency = {collection: Counter() for collection in COLLECTION_DIRS}
    total_documents_per_collection = {collection: 0 for collection in COLLECTION_DIRS}

    for collection, directory in COLLECTION_DIRS.items():
        for filename in os.listdir(directory):
            if not filename.endswith(".txt"):
                continue

            total_documents_per_collection[collection] += 1
            filepath = os.path.join(directory, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            tokens = filter_tokens(text)
            unique_tokens = set(tokens)

            collection_frequency[collection].update(tokens)
            document_frequency[collection].update(unique_tokens)

    all_tokens = set()
    for collection in COLLECTION_DIRS:
        all_tokens.update(collection_frequency[collection].keys())

    results: Dict[str, WordInfo] = {}

    for token in all_tokens:
        frequency_map = {
            collection: collection_frequency[collection].get(token, 0)
            for collection in COLLECTION_DIRS
        }

        winning_collection = max(frequency_map, key=frequency_map.get)
        df_in_winner = document_frequency[winning_collection].get(token, 0)
        num_docs_in_winner = total_documents_per_collection[winning_collection]

        idf_value = math.log((num_docs_in_winner + 1) / (df_in_winner + 1)) + 1

        results[token] = WordInfo(
            document_frequency=df_in_winner,
            idf=idf_value,
            collection_vote=winning_collection,
        )

    return results


WORD_STATS: Dict[str, WordInfo] = compute_collection_statistics()
