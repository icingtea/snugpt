import os
import math
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Optional
from pydantic import BaseModel

from models.chunks import CollectionEnum
from pos_tagger import pos_tagger, lemmatize


STOPWORDS = {
    "be",
    "is",
    "am",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "let",
    "make",
    "get",
    "got",
    "go",
    "gone",
    "see",
    "seen",
    "say",
    "said",
    "take",
    "took",
    "come",
    "came",
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
    filtered = [
        (token, pos)
        for token, pos in tagged_tokens
        if pos in ("NOUN", "VERB") and token.lower() not in STOPWORDS
    ]

    lemmatized = lemmatize(filtered)
    return [token.lower() for token, _ in lemmatized]


ROOT_DATA_DIR = "data/extracted"

COLLECTION_DIRS = {
    CollectionEnum.ACADEMICS: os.path.join(ROOT_DATA_DIR, "academics"),
    CollectionEnum.FACULTY: os.path.join(ROOT_DATA_DIR, "faculty"),
    CollectionEnum.STUDENTS: os.path.join(ROOT_DATA_DIR, "students"),
}

STATS_PATH = Path("data/cache/word_stats.json")


# def compute_collection_statistics() -> Dict[str, WordInfo]:
#     collection_frequency = {collection: Counter() for collection in COLLECTION_DIRS}
#     document_frequency = {collection: Counter() for collection in COLLECTION_DIRS}
#     total_documents_per_collection = {collection: 0 for collection in COLLECTION_DIRS}

#     for collection, directory in COLLECTION_DIRS.items():
#         if not os.path.exists(directory):
#             continue

#         for filename in os.listdir(directory):
#             if not filename.endswith(".txt"):
#                 continue

#             total_documents_per_collection[collection] += 1
#             filepath = os.path.join(directory, filename)

#             with open(filepath, "r", encoding="utf-8") as f:
#                 text = f.read()

#             tokens = filter_tokens(text)
#             unique_tokens = set(tokens)

#             collection_frequency[collection].update(tokens)
#             document_frequency[collection].update(unique_tokens)

#     all_tokens = set()
#     for collection in COLLECTION_DIRS:
#         all_tokens.update(collection_frequency[collection].keys())

#     results: Dict[str, WordInfo] = {}

#     for token in all_tokens:
#         frequency_map = {
#             collection: collection_frequency[collection].get(token, 0)
#             for collection in COLLECTION_DIRS
#         }

#         winning_collection = max(frequency_map, key=frequency_map.get)
#         df_in_winner = document_frequency[winning_collection].get(token, 0)
#         num_docs_in_winner = total_documents_per_collection[winning_collection]

#         idf_value = math.log((num_docs_in_winner + 1) / (df_in_winner + 1)) + 1

#         results[token] = WordInfo(
#             document_frequency=df_in_winner,
#             idf=idf_value,
#             collection_vote=winning_collection,
#         )

#     return results


def compute_collection_statistics() -> Dict[str, WordInfo]:
    collection_frequency = {collection: Counter() for collection in COLLECTION_DIRS}
    document_frequency = {collection: Counter() for collection in COLLECTION_DIRS}
    total_documents_per_collection = {collection: 0 for collection in COLLECTION_DIRS}

    for collection, directory in COLLECTION_DIRS.items():
        if not os.path.exists(directory):
            continue

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

    raw_idf_by_collection = {collection: {} for collection in COLLECTION_DIRS}

    for token in all_tokens:
        for collection in COLLECTION_DIRS:
            df = document_frequency[collection].get(token, 0)
            num_docs = total_documents_per_collection[collection]
            raw_idf = math.log((num_docs + 1) / (df + 1)) + 1
            raw_idf_by_collection[collection][token] = raw_idf

    normalized_idf_by_collection = {collection: {} for collection in COLLECTION_DIRS}

    for collection in COLLECTION_DIRS:
        values = list(raw_idf_by_collection[collection].values())
        if not values:
            continue

        min_idf = min(values)
        max_idf = max(values)
        range_idf = max_idf - min_idf

        for token, value in raw_idf_by_collection[collection].items():
            if range_idf == 0:
                normalized_value = 0.0
            else:
                normalized_value = (value - min_idf) / range_idf
            normalized_idf_by_collection[collection][token] = normalized_value

    results: Dict[str, WordInfo] = {}

    for token in all_tokens:
        frequency_map = {
            collection: collection_frequency[collection].get(token, 0)
            for collection in COLLECTION_DIRS
        }

        winning_collection = max(frequency_map, key=frequency_map.get)
        df_in_winner = document_frequency[winning_collection].get(token, 0)
        normalized_idf = normalized_idf_by_collection[winning_collection].get(
            token, 0.0
        )

        results[token] = WordInfo(
            document_frequency=df_in_winner,
            idf=normalized_idf,
            collection_vote=winning_collection,
        )

    return results


def save_word_stats(stats: Dict[str, WordInfo]):
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for token, info in stats.items():
        data[token] = {
            "document_frequency": info.document_frequency,
            "idf": info.idf,
            "collection_vote": info.collection_vote.value,
        }
    with STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_word_stats() -> Dict[str, WordInfo]:
    with STATS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    stats = {}
    for token, info in data.items():
        stats[token] = WordInfo(
            document_frequency=info["document_frequency"],
            idf=info["idf"],
            collection_vote=CollectionEnum(info["collection_vote"]),
        )
    return stats


if STATS_PATH.exists():
    WORD_STATS = load_word_stats()
else:
    print("Computing Collection Statistics")
    WORD_STATS = compute_collection_statistics()
    save_word_stats(WORD_STATS)
