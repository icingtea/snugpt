# graph_nodes.py
import os
import logging
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
from pymongo import MongoClient
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer
import openai
from langchain_core.messages import HumanMessage, AIMessage

from src.models.graph import GraphState
from src.models.chunks import CollectionEnum
from src.vocab.vocabulary import WORD_STATS, filter_tokens

load_dotenv()

logging.basicConfig(filename="session.log", filemode="w")
logger = logging.getLogger("applog")
logger.setLevel(logging.DEBUG)

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = ""

if not all([MONGODB_CONNECTION_STRING, OPENAI_API_KEY]):
    raise EnvironmentError("[ERROR] Missing required environment variables")

mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
openai_client = openai.OpenAI()
embedding_model = SentenceTransformer(EMBEDDING_MODEL) if EMBEDDING_MODEL else None


ACADEMICS_KEYWORDS = {
    "keywords": [],
    "schools": {
        "SOE": ["engineering", "engg", "soe", "school of engineering"],
        "SNS": ["science", "sns", "school of science", "natural sciences"],
        "SHSS": ["humanities", "shss", "school of humanities", "social sciences", "humanities and social sciences"],
        "SME": ["management", "sme", "school of management", "business", "business school"]
    },
    "departments": {
        "CSE": ["cs", "CS", "computer science", "cse", "CSE", "computer science and engineering"],
        "ECE": ["ee", "electrical engineering", "ece", "ECE", "electrical and computer engineering"],
        "MECH": ["mechanical engineering", "mech", "MECH"],
        "CHEM_ENG": ["chemical engineering", "chem eng", "CHEM_ENG"],
        "CIVIL": ["civil engineering", "civil", "CIVIL"],
        "MATH": ["math", "maths", "mathematics", "MATH"],
        "PHY": ["physics", "phy", "PHY"],
        "CHEM": ["chemistry", "chem", "CHEM"],
        "BIOTECH": ["biotechnology", "biotech", "BIOTECH"],
        "ECO": ["economics", "eco", "ECO"],
        "ENG": ["english", "eng", "ENG"]
    }
}

FACULTY_KEYWORDS = {
    "keywords": [],
    "schools": {
        "SOE": ["engineering", "engg", "soe", "school of engineering"],
        "SNS": ["science", "sns", "school of science", "natural sciences"],
        "SHSS": ["humanities", "shss", "school of humanities", "social sciences", "humanities and social sciences"],
        "SME": ["management", "sme", "school of management", "business", "business school"]
    },
    "departments": {
        "CSE": ["cs", "CS", "computer science", "cse", "CSE", "computer science and engineering"],
        "ECE": ["ee", "electrical engineering", "ece", "ECE", "electrical and computer engineering"],
        "MECH": ["mechanical engineering", "mech", "MECH"],
        "CHEM_ENG": ["chemical engineering", "chem eng", "CHEM_ENG"],
        "CIVIL": ["civil engineering", "civil", "CIVIL"],
        "MATH": ["math", "maths", "mathematics", "MATH"],
        "PHY": ["physics", "phy", "PHY"],
        "CHEM": ["chemistry", "chem", "CHEM"],
        "BIOTECH": ["biotechnology", "biotech", "BIOTECH"],
        "ECO": ["economics", "eco", "ECO"],
        "ENG": ["english", "eng", "ENG"]
    }
}


STUDENTS_KEYWORDS = {
    "keywords": ["academic calendar", "exam schedule", "course registration", "hostel", "library"]
}

MENU_KEYWORDS = {
    "keywords": []
}

COLLECTION_INDEX_MAP = {
    CollectionEnum.ACADEMICS: "academics_vector_index",
    CollectionEnum.FACULTY: "faculty_vector_index", 
    CollectionEnum.STUDENTS: "vector_index",
    CollectionEnum.MENU: None
}


def build_filter_from_keywords(prompt_lower: str, keyword_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    filter_conditions = []

    schools_dict = keyword_config.get("schools", {})
    for actual_school, variations in schools_dict.items():
        if any(variation in prompt_lower for variation in variations):
            filter_conditions.append({"schools": actual_school})

        elif actual_school in prompt_lower:
            filter_conditions.append({"schools": actual_school})

    departments_dict = keyword_config.get("departments", {})
    for actual_dept, variations in departments_dict.items():

        if any(variation in prompt_lower for variation in variations):
            filter_conditions.append({"departments": actual_dept})

        elif actual_dept in prompt_lower:
            filter_conditions.append({"departments": actual_dept})
    
    if filter_conditions:
        return {"$or": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]
    
    return None


def keyword_router(state: GraphState) -> Dict[str, Any]:
    """Route based on explicit keywords in the prompt and build filters."""
    prompt = state["prompt"]
    if not prompt:
        return {"collections": [], "filter": None}
    
    prompt_lower = prompt.lower()
    collections_to_search = []
    filter_condition = None
    
    academics_filter = build_filter_from_keywords(prompt_lower, ACADEMICS_KEYWORDS)
    faculty_filter = build_filter_from_keywords(prompt_lower, FACULTY_KEYWORDS)
    
    if academics_filter or faculty_filter:
        collections_to_search = [CollectionEnum.ACADEMICS, CollectionEnum.FACULTY]

        if academics_filter and faculty_filter:
            filter_condition = {"$or": [academics_filter, faculty_filter]}
        else:
            filter_condition = academics_filter or faculty_filter
    
    elif any(kw in prompt_lower for kw in STUDENTS_KEYWORDS["keywords"]):
        collections_to_search = [CollectionEnum.STUDENTS]
        filter_condition = None

    elif any(kw in prompt_lower for kw in MENU_KEYWORDS["keywords"]):
        collections_to_search = [CollectionEnum.MENU]
        filter_condition = None
    
    if collections_to_search:
        state_change = {
            "collections": collections_to_search,
            "filter": filter_condition
        }
        logger.info(f"[KEYWORD ROUTER] Collections: {collections_to_search}, Filter: {filter_condition}")
        return state_change

    state_change = {"collections": [], "filter": None}
    logger.info(f"[KEYWORD ROUTER] No keyword match")
    return state_change


def vocab_voter(state: GraphState) -> Dict[str, Any]:
    prompt = state["prompt"]
    if not prompt:
        return {"collections": [CollectionEnum.STUDENTS], "filter": None}
    
    tokens = filter_tokens(prompt)
    
    if not tokens:
        state_change = {"collections": [CollectionEnum.STUDENTS], "filter": None}
        logger.info(f"[VOCAB VOTER] No valid tokens, defaulting to STUDENTS")
        return state_change
    
    votes: Dict[CollectionEnum, float] = {
        CollectionEnum.ACADEMICS: 0.0,
        CollectionEnum.FACULTY: 0.0,
        CollectionEnum.STUDENTS: 0.0,
        CollectionEnum.MENU: 0.0,
    }
    
    for token in tokens:
        word_info = WORD_STATS.get(token)
        if word_info:
            votes[word_info.collection_vote] += word_info.idf or 0.0

    if votes:
        winner = max(votes.items(), key=lambda x: x[1])
        prompt_lower = prompt.lower()
        filter_condition = None
        collections_to_search = [winner[0]]

        if winner[0] in [CollectionEnum.ACADEMICS, CollectionEnum.FACULTY]:
            collections_to_search = [CollectionEnum.ACADEMICS, CollectionEnum.FACULTY]

            filter_academics = build_filter_from_keywords(prompt_lower, ACADEMICS_KEYWORDS)
            filter_faculty = build_filter_from_keywords(prompt_lower, FACULTY_KEYWORDS)

            if filter_academics and filter_faculty:
                filter_condition = {"$or": [filter_academics, filter_faculty]}
            else:
                filter_condition = filter_academics or filter_faculty

        state_change = {"collections": collections_to_search, "filter": filter_condition}
        logger.info(f"[VOCAB VOTER] Votes: {votes}, Winner: {winner[0]}, Collections: {collections_to_search}, Filter: {filter_condition}")
        return state_change
    
    state_change = {"collections": [CollectionEnum.STUDENTS], "filter": None}
    logger.info(f"[VOCAB VOTER] No votes, defaulting to STUDENTS")
    return state_change


def vector_search(state: GraphState) -> Dict[str, Any]:
    if not embedding_model:
        return {"context": [], "error": "[ERROR] Embedding model not configured"}
    
    collections = state.get("collections", [])
    prompt = state["prompt"]
    filter_condition = state.get("filter")
    
    if not collections or not prompt:
        return {"context": [], "error": "[ERROR] Missing collections or prompt"}

    all_context_docs = []
    
    for collection in collections:
        try:
            db = mongo_client["snugpt"]
            mongo_collection: Collection = db[collection.value]
            
            if collection == CollectionEnum.MENU:
                if filter_condition:
                    results = mongo_collection.find(filter_condition)
                else:
                    results = mongo_collection.find()
                
                context_docs = [doc.get("document", doc.get("text", "")) for doc in results.limit(10)]
                all_context_docs.extend(context_docs)
                logger.info(f"[VECTOR SEARCH] Found {len(context_docs)} menu items from {collection.value}")
                continue

            search_index = COLLECTION_INDEX_MAP.get(collection)
            
            if not search_index:
                logger.warning(f"[VECTOR SEARCH] No vector index configured for {collection.value}")
                continue

            prompt_embedding = embedding_model.encode([prompt]).tolist()[0]

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": search_index,
                        "path": "embedding",
                        "queryVector": prompt_embedding,
                        "exact": True,
                        "limit": 15,
                        "filter": filter_condition,
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "text": 1,
                        "document": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            
            results = mongo_collection.aggregate(pipeline)
            context_docs = []
            for doc in results:
                content = doc.get("document") or doc.get("text", "")
                score = float(doc.get("score", 0))
                if score > 0.5:
                    context_docs.append(content)
            
            all_context_docs.extend(context_docs)
            logger.info(f"[VECTOR SEARCH] Found {len(context_docs)} relevant chunks from {collection.value} with filter: {filter_condition}")
            
        except Exception as e:
            logger.error(f"[VECTOR SEARCH] Error searching {collection.value}: {e}")
    
    state_change = {"context": all_context_docs, "error": None}
    logger.info(f"[VECTOR SEARCH] Total found {len(all_context_docs)} context docs from {len(collections)} collections")
    return state_change


def chat_response(state: GraphState) -> Dict[str, Any]:
    client = openai_client
    prompt = state["prompt"]
    context = state.get("context", [])
    memory = state.get("memory", [])
    
    system_prompt = """
    You are SNUGPT, a helpful assistant for Shiv Nadar University students, faculty, and staff.
    Use the provided context to answer questions accurately and precisely.
    
    If you don't have enough information to answer, say so clearly.
    Do not mention "context" or "filters" in your response.
    Be concise and factual.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"MESSAGE HISTORY: {memory}\nCONTEXT: {context}\nUSER PROMPT: {prompt}"
                },
            ],
        )
        
        answer = response.choices[0].message.content
        
        state_change = {
            "response": answer,
            "memory": [
                HumanMessage(content=prompt),
                AIMessage(content=answer),
            ],
            "error": None,
        }
        logger.info(f"[CHAT RESPONSE] Generated response")
        return state_change
        
    except Exception as e:
        state_change = {
            "response": None,
            "error": f"[ERROR] Failed to get chat response: {e}",
        }
        logger.error(f"[CHAT RESPONSE] {e}")
        return state_change


def error_response(state: GraphState) -> Dict[str, Any]:
    state_change = {"response": state.get("error", "An unknown error occurred")}
    logger.info(f"[ERROR RESPONSE] {state_change}")
    return state_change


def error_check(state: GraphState) -> bool:
    return state.get("error") is not None


def needs_vocab_vote(state: GraphState) -> bool:
    return len(state.get("collections", [])) == 0