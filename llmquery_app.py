import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import re

# enable to print statements
print_debug = False

def load_fuzzy_rules():
    """
    # load json for fuzzy rules
    """
    with open("data/fuzzy_rules.json", "r") as f:
        return json.load(f)

FUZZY_RULES = load_fuzzy_rules()


def expand_fuzzy_terms(query):
    """
    check for synonyms or letter casing
    
    :param query: user query from streamlit
    """
    query_lower = query.lower()

    detected = []

    for key, rule in FUZZY_RULES.items():
        # Check canonical term - lower case
        if key in query_lower:
            detected.append(key)
            continue

        # Check synonyms
        for syn in rule["synonyms"]:
            if syn in query_lower:
                detected.append(key)
                break

    return detected

# ---------------------------------------------------------
# LLM INTERPRETER (Ollama Llama3)
# ---------------------------------------------------------
def interpret_query_with_llm(query):
    """
    convert the query into filter (json format - dict) by asking prompt to LLM model
    
    :param query: user query from streamlit
    """
    fuzzy_hits = expand_fuzzy_terms(query)

    fuzzy_instructions = ""
    for term in fuzzy_hits:
        fuzzy_instructions += f'"{term}" → {json.dumps(FUZZY_RULES[term]["filters"])}\n'

    prompt = f"""
You are a query interpreter for a KITTI dataset explorer.

Your ONLY job is to output a JSON object. 
No explanations. No markdown. No commentary. No notes.

The JSON must contain exactly:
- "filters": a dict mapping metadata fields to numeric conditions
- "semantic_query": a text query for semantic search (or null)

Valid metadata fields:
- num_cars
- num_pedestrians
- num_cyclists
- max_occlusion
- max_truncation

Valid operators: ">", ">=", "<", "<="

Never include empty filter objects. Only include filters that have valid numeric conditions.

Fuzzy → Numeric rules (derived from fuzzy_rules.json):
{fuzzy_instructions}

User query: "{query}"

Return ONLY valid JSON. Nothing else.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        # get the raw response
        raw = response.json().get("response", "").strip()
        if print_debug:
            print("\nRAW LLM OUTPUT:\n", raw)

        # Extract JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM output")

        # get only json 
        json_text = match.group(0)

        # Replace Mongo-style operators
        json_text = json_text.replace("$gt", ">")
        json_text = json_text.replace("$lt", "<")
        json_text = json_text.replace("$gte", ">=")
        json_text = json_text.replace("$lte", "<=")

        # return query as dict (query -> filters)
        parsed = json.loads(json_text)
        if print_debug:
            print("\nPARSED JSON:\n", parsed)
        return parsed

    except Exception as e:
        print("LLM error:", e)
        return {"filters": {}, "semantic_query": query}


# ---------------------------------------------------------
# LOAD RAG COMPONENTS
# ---------------------------------------------------------
@st.cache_resource
def load_rag():
    """
    load RAG components (labels file doc, faiss index, transformer embedded model)
    """
    with open("data/kitti_docs.json", "r") as f:
        docs = json.load(f)

    index = faiss.read_index("data/kitti_index.faiss")

    with open("data/embedding_model.txt", "r") as f:
        model_name = f.read().strip()

    model = SentenceTransformer(model_name)

    return docs, index, model


docs, index, emb_model = load_rag()


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title(" KITTI RAG Explorer (Ollama-Powered)")
st.write("Ask natural language questions about KITTI scenes, objects, occlusion, truncation, or counts.")

# USER UI
query = st.text_input("Enter your query")
top_k = st.slider("Number of results", 1, 10, 5)

if query:

    # ---------------------------------------------------------
    # STEP 1 — LLM INTERPRETATION (query - filters)
    # ---------------------------------------------------------
    parsed = interpret_query_with_llm(query)
    filters = parsed.get("filters", {})
    semantic_query = parsed.get("semantic_query", None)

    st.markdown("### LLM Interpretation")
    st.json(parsed)

    # ---------------------------------------------------------
    # STEP 2 — APPLY NUMERIC FILTERS
    # ---------------------------------------------------------
    filtered_docs = docs

    for field, condition in filters.items():

        # Skip empty or malformed conditions 
        if not isinstance(condition, dict) or len(condition) == 0:
            continue

        # get operation sign and threshold value
        op, value = list(condition.items())[0]

        if op == ">":
            filtered_docs = [d for d in filtered_docs if d[field] > value]
        elif op == ">=":
            filtered_docs = [d for d in filtered_docs if d[field] >= value]
        elif op == "<":
            filtered_docs = [d for d in filtered_docs if d[field] < value]
        elif op == "<=":
            filtered_docs = [d for d in filtered_docs if d[field] <= value]

    # If numeric filters matched something → show those results
    if filters and len(filtered_docs) > 0:
        st.success(f"Found {len(filtered_docs)} frames matching numeric filters")

        # Sort by the most relevant field
        sort_field = list(filters.keys())[0]
        filtered_docs = sorted(filtered_docs, key=lambda d: d[sort_field], reverse=True)

        # display filtered result images
        for d in filtered_docs[:top_k]:
            st.subheader(f"Frame {d['id']}")
            st.write(d["summary_text"])
            st.image(d["image_path"])
        st.stop()

    # if not label file is detected
    if filters and len(filtered_docs) == 0:
        st.warning("No frames matched all numeric filters. Falling back to semantic search.")

    # ---------------------------------------------------------
    # STEP 3 — SEMANTIC SEARCH FALLBACK
    # ---------------------------------------------------------
    if semantic_query is None:
        semantic_query = query

    # query from sentence transformer
    q_emb = emb_model.encode([semantic_query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, k=top_k)

    st.markdown("### Semantic Search Results")

    # display result images
    for idx in I[0]:
        doc = docs[idx]
        st.subheader(f"Frame {doc['id']}")
        st.write(doc["summary_text"])
        st.image(doc["image_path"])
