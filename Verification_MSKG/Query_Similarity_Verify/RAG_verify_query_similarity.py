# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: KnowledgeGraph_RAG
@File Name: .py
@Software: Python
@Time: Feb/2025
@Author: Yufei Quan, Rui Xu
@Contact: yufeiq@kth.se, rxu@kth.se
@Version: 0.1.1
@Description: Node-based RAG pipeline with Gemini embeddings and pickle-based caching, and use embedding for matching.
              Optimized cypher query method and user question embedding method.
              allow to ask question reach to different depth relatives.
              Dynamically modify query depth using question semantic diversity and sentence length.
"""

import pickle
import re
import numpy as np
from neo4j import GraphDatabase
import google.generativeai as genai
from porlog import configuration
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import json


# Configure Gemini API
genai.configure(api_key=configuration.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


class Neo4jRAGSystem:
    def __init__(self, uri, user, password, cache_file="embedding_cache.pkl"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=60)
        self.cache_file = cache_file
        self.embedding_cache = self.load_cache()

    def close(self):
        self.driver.close()

    def load_cache(self):
        """Load embedding cache from pickle file."""
        try:
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}

    def save_cache(self):
        """Save embedding cache to pickle file."""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.embedding_cache, f)

    def cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        Ensure both vectors are NumPy arrays before computation.
        """
        if isinstance(vec1, str):
            try:
                vec1 = json.loads(vec1)
            except json.JSONDecodeError:
                print(f"[ERROR] Failed to decode embedding: {vec1}")
                return 0.0
        if isinstance(vec2, str):
            try:
                vec2 = json.loads(vec2)
            except json.JSONDecodeError:
                print(f"[ERROR] Failed to decode embedding: {vec2}")
                return 0.0

        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def update_embeddings(self):
        """Update missing embeddings for nodes in the Neo4j database."""
        with self.driver.session() as session:
            cypher_query = """
            MATCH (n)
            WHERE n.Node_Type IN ['Business', 'Certification', 'Industry', 'Manufactor', 'Material']
              AND n.embedding IS NULL
            RETURN n.Id AS id, n.Name AS name, elementId(n) AS node_id
            """
            records = session.run(cypher_query)

            for record in records:
                text = record['name']  # Use 'name' field instead of 'title'+'body_markdown'
                embedding = self.get_embedding(text)
                if embedding:
                    update_query = """
                    MATCH (n) WHERE elementId(n) = $node_id
                    SET n.embedding = $embedding
                    """
                    session.run(update_query, node_id=record['node_id'], embedding=embedding)
            print("Embeddings updated successfully.")

    def get_embedding(self, text):
        """Generate embedding using Gemini API or fetch from cache."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            embedding = response["embedding"]
            self.embedding_cache[text] = embedding
            self.save_cache()
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def extract_keywords(self, query_text, similarity_threshold=0.4):
        """
            1. Calculate the semantic similarity between each word and the question as a whole.
            2. Dynamically filter irrelevant words without the need for a stop word list.
        """
        words = re.findall(r'[a-zA-Z0-9._-]+', query_text)

        # calaculate whole question's Embedding
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            return []

        # Calcualate every words' similarity and embedding.
        keyword_weights = []
        for word in words:
            word_embedding = self.get_embedding(word)
            if word_embedding is None:
                continue
            similarity = cosine_similarity([word_embedding], [query_embedding])[0][0]
            keyword_weights.append((word, similarity))

        # Indexing
        keyword_weights.sort(key=lambda x: x[1], reverse=True)
        keywords = [word for word, similarity in keyword_weights if similarity >= similarity_threshold]

        return keywords

    def compute_semantic_diversity(self, query_text):
        """
        Compute the semantic diversity of a given query by analyzing the variance in word embeddings.
        A higher variance indicates a more diverse semantic meaning in the query.
        """
        stopwords = {"what", "which", "how", "is", "in", "the", "and", "does", "to", "do"}
        words = [word for word in re.findall(r'\b\w+\b', query_text.lower())
                 if word not in stopwords]

        if len(words) < 5:
            return 0.4

        # Retrieve embeddings for up to 10 words.
        embeddings = np.array([self.get_embedding(word) for word in words[:10]])

        # Apply PCA to reduce dimensionality and analyze variance.
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        variance = np.sum(np.var(reduced, axis=0))

        # Normalize variance using a maximum expected variance value.
        max_variance = 4.0
        return min(variance / max_variance, 1.0)

    def determine_query_depth(self, query_text):
        """
        Determine the appropriate query depth based on semantic and syntactic complexity.
        """
        semantic_score = self.compute_semantic_diversity(query_text)
        syntax_score = self.compute_syntactic_complexity(query_text)

        # Weighted combination of semantic and syntactic complexity.
        combined_score = 0.6 * semantic_score + 0.4 * syntax_score

        if combined_score < 0.3:
            depth = 1
        elif 0.3 <= combined_score < 0.6:
            depth = 2
        elif 0.6 <= combined_score < 0.8:
            depth = 3
        else:
            depth = 4

        if re.search(r'\b(indirect|through|chain|subsidiary)\b', query_text, re.I):
            depth = max(depth, 3)

        return depth

    def compute_syntactic_complexity(self, query_text):
        """
        Compute syntactic complexity based on sentence structure.
        """
        words = query_text.split()

        # Normalize sentence length to a max of 15 words.
        length_score = min(len(words) / 15, 1.0)

        wh_words = {"what", "which", "how", "why", "who", "where", "analyze", "explain", "describe"}
        wh_score = min(sum(1 for word in words if word.lower() in wh_words) / 2, 1.0)

        # Count clauses
        clause_count = len(re.findall(r',|;| but | however | although ', query_text))
        clause_score = min(clause_count / 2, 1.0)

        # Compute weighted complexity score.
        return (length_score * 0.5 + wh_score * 0.3 + clause_score * 0.2)

    def analyze_query_intent_with_LLM(self, query_text):
        """
        Parse query intent through LLM to extract target entities, relationships, and filter conditions
        """
        system_prompt = """
        You are an AI assistant specialized in knowledge graph queries. 
        Given a user question, extract the key elements required to construct a Neo4j Cypher query:

        1. The main entity type(s) being queried (choose from: Business, Certification, Industry, Manufactor, Material).
        2. The relationship types needed to answer the question (choose from: Belong, Certify, Process, Sub_Bussiness, Sub_Industry, Work_on).
        3. Any specific filtering conditions (e.g., requiring a specific certification like ISO9001).

        Return a JSON response in the following format:
        {
            "entities": ["Manufactor"],
            "relationships": ["Certify", "Process"],
            "filters": {"Certification": "ISO9001"}
        }
        """

        response = model.generate_content(system_prompt + "\nUser question: " + query_text)

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"entities": ["Manufactor"], "relationships": [], "filters": {}}


    def retrieve_relevant_entities(self, query_text, top_k=3, use_similarity=False):
        """
        Retrieves relevant entities using either keyword matching or embedding-based similarity search.
        """
        if use_similarity:
            return self._retrieve_by_similarity(query_text, top_k, similarity_threshold=0.7)
        else:
            return self._retrieve_by_keywords(query_text, top_k)

    def _retrieve_by_keywords(self, query_text, top_k):
        """
        Retrieves entities using keyword matching, supporting depth expansion.
        Dynamically detects available labels to avoid using non-existent ones.
        """

        depth = self.determine_query_depth(query_text)

        keywords = self.extract_keywords(query_text)
        if not keywords:
            print(f"[WARNING] No keywords extracted from `{query_text}`.")
            return []

        print(f"[INFO] Extracted keywords for `{query_text}`: {keywords}")

        with self.driver.session() as session:
            result = session.run("CALL db.labels()")
            existing_labels = {record["label"] for record in result}

        valid_labels = {"Business", "Certification", "Industry", "Manufactor", "Material"}
        selected_labels = valid_labels.intersection(existing_labels)

        if not selected_labels:
            print("[ERROR] No valid labels found in the database.")
            return []

        label_conditions = " OR ".join([f"n:{label}" for label in selected_labels])

        cypher_query = f"""
        MATCH p = (n)-[r*1..{depth}]-(m)
        WHERE ({label_conditions}) AND any(kw IN $keywords WHERE toLower(n.Name) CONTAINS kw)
        RETURN DISTINCT
            n.Id AS node1_id, n.Name AS node1_name, labels(n) AS node1_type,
            [rel IN relationships(p) | type(rel)] AS relations,
            [x IN nodes(p) | {{id: x.Id, name: x.Name, type: labels(x)}}] AS connected_nodes
        LIMIT {top_k}
        """

        with self.driver.session() as session:
            records = session.run(cypher_query, keywords=[kw.lower() for kw in keywords])

            entity_graphs = []
            for record in records:
                entity_graphs.append({
                    "node1": {
                        "id": record["node1_id"],
                        "name": record["node1_name"],
                        "type": record["node1_type"]
                    },
                    "relations": record["relations"],
                    "connected_nodes": record["connected_nodes"]
                })

        print(f"[INFO] Keyword search completed: `{query_text}`, retrieved {len(entity_graphs)} results.")
        return entity_graphs

    def _retrieve_by_similarity(self, query_text, top_k=5, similarity_threshold=0.7):
        """
        Retrieve relevant entities by combining keyword and embedding similarity results.
        Supports multi-hop depth expansion for both keyword and fallback similarity.
        """

        depth = self.determine_query_depth(query_text)

        # Step 1: Keyword-based results
        keyword_results = self._retrieve_by_keywords(query_text, top_k=100)
        existing_ids = {entry["node1"]["id"] for entry in keyword_results}
        print(f"[INFO] Found {len(keyword_results)} results from keyword search.")

        # Step 2: Similarity fallback
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            print("[WARNING] No embedding generated for query.")
            return keyword_results

        similarity_candidates = []
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                RETURN n.Id AS id, n.Name AS name, labels(n) AS type, n.embedding AS embedding
                LIMIT 200
            """)

            for record in result:
                if record["id"] in existing_ids:
                    continue

                sim = self.cosine_similarity(query_embedding, record["embedding"])
                if sim >= similarity_threshold:
                    similarity_candidates.append({
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "similarity": sim
                    })

        similarity_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        selected_similar = similarity_candidates[:top_k]

        for item in selected_similar:
            with self.driver.session() as session:
                result = session.run("""
                MATCH p = (n)-[r*1..{depth}]-(m)
                WHERE n.Id = $id
                RETURN DISTINCT
                    n.Id AS node1_id, n.Name AS node1_name, labels(n) AS node1_type,
                    [rel IN relationships(p) | type(rel)] AS relations,
                    [x IN nodes(p) | {{id: x.Id, name: x.Name, type: labels(x)}}] AS connected_nodes
                LIMIT 1
                """, id=item["id"])

                for record in result:
                    keyword_results.append({
                        "node1": {
                            "id": record["node1_id"],
                            "name": record["node1_name"],
                            "type": record["node1_type"]
                        },
                        "relations": record["relations"],
                        "connected_nodes": record["connected_nodes"]
                    })

        print(f"[INFO] Added {len(selected_similar)} additional results from similarity matching.")
        return keyword_results

    def generate_answer(self, user_question, relevant_entities):
        """
        Generate a structured answer using Gemini LLM based on the retrieved knowledge context.
        The prompt instructs the LLM to logically deduce all possible relationships based on the provided entities,
        regardless of whether direct information exists. It must produce a complete structured answer section
        containing all plausible triples and then detailed process reasoning.
        """
        # If no relevant entities are found, instruct the LLM to hypothesize based solely on user question.
        if not relevant_entities:
            fallback = ("insufficient data", "hypothesized", "insufficient data")
            print("[LLM Fallback] Using fallback triple:", fallback)
            return [fallback]

        # Extract keywords for context
        keywords = self.extract_keywords(user_question)

        # Construct prompt with instructions to use the provided knowledge context and perform thorough reasoning.
        input_text = (
            "You are an expert AI assistant specialized in knowledge graph analysis.\n"
            "Your task is to answer the following question based on the provided multi-hop knowledge context.\n"
            "IMPORTANT: Even if the provided context is limited, you MUST produce a structured answer section "
            "that includes ALL logically deduced relationships in the format:\n"
            "    (Entity A)-[:Relationship]->(Entity B)\n"
            "Follow this with a detailed process reasoning section that explains your inference step-by-step.\n"
            "Do NOT output placeholder phrases such as 'No answer' or 'insufficient data'.\n\n"
            "### Extracted Keywords:\n" + ", ".join(keywords) + "\n\n"
            "### Knowledge Context:\n"
        )

        # Append retrieved entities' information to context
        for entity in relevant_entities:
            node = entity.get("node1", {})
            node_name = node.get("name", "unknown")
            node_types = ", ".join(node.get("type", []))
            rels = ", ".join(entity.get("relations", []))
            connected = ", ".join([c.get("name", "unknown") for c in entity.get("connected_nodes", [])])
            input_text += f"- Entity: {node_name} (Type: {node_types}); Relations: {rels}; Connected: {connected}\n"

        # Add user question and instruction to produce structured answer.
        input_text += (
            "\n### User Question:\n" + user_question + "\n\n"
            "### Final Answer:\n"
            "Based on the knowledge context above, provide ALL plausible structured triples "
            "in the format: (Entity)-[:Relationship]->(Entity). Then, provide your detailed reasoning."
        )

        # Call the LLM to generate the answer.
        llm_response = self.generate_with_llm(input_text)
        print("\n[LLM Response]\n" + llm_response)

        # Use regex to extract all structured triples from the entire LLM response.
        pattern = r"\(([^)]+)\)-\[:([^\]]+)\]->\(([^)]+)\)"
        extracted_triples = []
        for match in re.finditer(pattern, llm_response):
            subject = match.group(1).strip().lower()
            relation = match.group(2).strip().lower()
            obj = match.group(3).strip().lower()
            # Only add if triple components are not placeholders.
            if subject in ["no answer", "insufficient data"] or obj in ["no answer", "insufficient data"]:
                continue
            extracted_triples.append((subject, relation, obj))

        # If no triples extracted, warn and return fallback triple.
        if not extracted_triples:
            print("[WARNING] No structured triples extracted from LLM response. Returning fallback triple.")
            extracted_triples.append(("unknown", "unknown", "unknown"))

        return extracted_triples

    def generate_with_llm(self, input_text):
        """
        Generate response using Gemini LLM with output cleaning.
        """
        try:
            response = model.generate_content(input_text)
            return self.clean_response(response.text)
        except Exception as e:
            return "Error: " + str(e)

    def clean_response(self, text):
        """
        Remove Markdown formatting and redundant symbols from LLM output.
        """
        text = text.replace("*", "").replace("###", "").strip()
        return "\n".join([line.strip() for line in text.split("\n") if line.strip()])


    def rag_pipeline(self, user_question):
        """Execute the RAG pipeline with embedding-based retrieval and in-context learning."""
        relevant_entities = self.retrieve_relevant_entities(user_question)
        return self.generate_answer(user_question, relevant_entities)


def main():
    rag_system = Neo4jRAGSystem(
        uri=configuration.NEO4J_URI,
        user=configuration.NEO4J_USER,
        password=configuration.NEO4J_PASSWORD
    )
    try:
        print("Updating embeddings...")
        rag_system.update_embeddings()
        user_input = input("Enter your question: ")
        answer = rag_system.rag_pipeline(user_input)
        print("\nFinal Answer:", answer)
    finally:
        rag_system.close()


if __name__ == "__main__":
    main()