# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: KnowledgeGraph_RAG
@File Name: main_rag_dynamatic_masked.py
@Software: Python
@Time: Feb/2025
@Author: Yufei Quan, Rui Xu
@Contact: yufeiq@kth.se, rxu@kth.se
@Version: 0.2.1
@Description: Node-based RAG pipeline with Gemini embeddings and pickle-based caching, and use embedding for matching.
              Optimized cypher query method and user question embedding method.
              allow to ask question reach to different depth relatives.
              Dynamically modify query depth using question semantic diversity and sentence length.
"""

import pickle
import re
import time
import numpy as np
from neo4j import GraphDatabase
import google.generativeai as genai
from main_RAG import configuration
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import spacy
import json

# Configure Gemini API
genai.configure(api_key=configuration.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


class Neo4jRAGSystem:
    def __init__(self, uri, user, password, cache_file="embedding_cache.pkl"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=60)
        self.cache_file = cache_file
        self.embedding_cache = self.load_cache()
        self.nlp = spacy.load("en_core_web_sm")

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
              AND n.masked = 1
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
            depth = max(depth, 4)

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

    def retrieve_relevant_entities(self, query_text, top_k=5, similarity_threshold=0.7):
        """
        First tries to retrieve relevant entities using keyword-based Cypher graph search (with depth).
        If that fails or returns no results, fallback to embedding-based similarity retrieval (with depth).
        Returns enriched entity subgraphs for downstream reasoning.
        """

        depth = self.determine_query_depth(query_text)

        # Keyword-based retrieval
        keywords = self.extract_keywords(query_text)
        if not keywords:
            print(f"[WARNING] No keywords extracted from `{query_text}`.")
            keywords = []

        print(f"[INFO] Extracted keywords for `{query_text}`: {keywords}")

        # Fetch valid node labels from DB
        with self.driver.session() as session:
            result = session.run("CALL db.labels()")
            existing_labels = {record["label"] for record in result}
        valid_labels = {"Business", "Certification", "Industry", "Manufactor", "Material"}
        selected_labels = valid_labels.intersection(existing_labels)

        if not selected_labels:
            print("[ERROR] No valid labels found in the database.")
            return []

        label_conditions = " OR ".join([f"n:{label}" for label in selected_labels])

        # Keyword search query with depth
        cypher_query = f"""
        MATCH p = (n)-[r*1..{depth}]-(m)
        WHERE ({label_conditions}) AND any(kw IN $keywords WHERE toLower(n.Name) CONTAINS kw) AND n.masked = 1
        RETURN DISTINCT
            n.Id AS node1_id, n.Name AS node1_name, labels(n) AS node1_type,
            [rel IN relationships(p) | type(rel)] AS relations,
            [x IN nodes(p) | {{id: x.Id, name: x.Name, type: labels(x)}}] AS connected_nodes
        LIMIT {top_k}
        """

        with self.driver.session() as session:
            keyword_records = session.run(cypher_query, keywords=[kw.lower() for kw in keywords])
            keyword_results = []
            existing_ids = set()

            for record in keyword_records:
                keyword_results.append({
                    "node1": {
                        "id": record["node1_id"],
                        "name": record["node1_name"],
                        "type": record["node1_type"]
                    },
                    "relations": record["relations"],
                    "connected_nodes": record["connected_nodes"]
                })
                existing_ids.add(record["node1_id"])

        if keyword_results:
            print(f"[INFO] Keyword-based search found {len(keyword_results)} results.")
            return keyword_results

        print("[INFO] No keyword-based match found. Falling back to similarity-based retrieval...")

        # Similarity fallback if keyword search fails
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            print("[WARNING] No embedding generated for query.")
            return []

        similarity_candidates = []
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.embedding IS NOT NULL AND n.masked = 1
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

        similarity_results = []
        for item in selected_similar:
            with self.driver.session() as session:
                result = session.run("""
                MATCH p = (n)-[r*1..{depth}]-(m)
                WHERE n.Id = $id AND n.masked = 1
                RETURN DISTINCT
                    n.Id AS node1_id, n.Name AS node1_name, labels(n) AS node1_type,
                    [rel IN relationships(p) | type(rel)] AS relations,
                    [x IN nodes(p) | {{id: x.Id, name: x.Name, type: labels(x)}}] AS connected_nodes
                LIMIT 1
                """, id=item["id"])

                for record in result:
                    similarity_results.append({
                        "node1": {
                            "id": record["node1_id"],
                            "name": record["node1_name"],
                            "type": record["node1_type"]
                        },
                        "relations": record["relations"],
                        "connected_nodes": record["connected_nodes"]
                    })

        print(f"[INFO] Similarity fallback found {len(similarity_results)} additional results.")
        return similarity_results

    def generate_answer(self, user_question, relevant_entities):
        """
        Generate an answer using Gemini LLM.
        If no direct matches are found, infer an answer based on the most similar nodes.
        """
        if not relevant_entities:
            return "I couldn't find a direct answer, but here's a guess based on the most relevant information I have."

        keywords = self.extract_keywords(user_question)

        input_text = f"""You are an expert AI assistant specializing in knowledge graph analysis.
        Your objective is to **always generate a structured knowledge graph answer first**, regardless of data availability.

        ## **Critical Instructions**:
           **Always start with structured knowledge graph output** in plain text (No JSON, No code blocks).
           - Correct format:
             (Entity A)-[:Relationship]->(Entity B)
             (Entity C)-[:Relationship]->(Entity D)
           - **Do NOT wrap the answer in ` ```json ` or any other code block.**
           - **Do NOT use `"subject": ..., "predicate": ..., "object": ... ` JSON format.**

          **Never include explanatory text before the structured output.**
           - **Do not write**: "Given the lack of direct information..."
           - **Do not use**: "Here's the structured output:"
           - **Directly present the structured knowledge graph.**

          **If exact matches exist, return them immediately.**
           - Use retrieved knowledge graph data directly.

           **If no exact matches exist, infer plausible relationships using entity similarity.**
           - **Never state that inference is being used—just present the structured answer.**

           **After the structured knowledge graph, provide a reasoning section.**
           - Introduce it with:
             ```
             ### Reasoning Process
             ```
           - Explain how relationships were inferred, **but only after presenting the structured answer.**

        ## **Extracted Keywords**:
        {", ".join(keywords)}

        ### Knowledge Context:
        Below are the retrieved entities and their multi-hop relationships:
        """

        for entity in relevant_entities:
            input_text += f"- **Entity:** {entity['node1']['name']} ({', '.join(entity['node1']['type'])})\n"
            input_text += f"  ├── **Relations:** {', '.join(entity['relations'])}\n"

            # Dealing the connection of muti-depth.
            industry_nodes = [node for node in entity["connected_nodes"] if "Industry" in node["type"]]
            cert_nodes = [node for node in entity["connected_nodes"] if "Certification" in node["type"]]
            manufactor_nodes = [node for node in entity["connected_nodes"] if "Manufactor" in node["type"]]
            material_nodes = [node for node in entity["connected_nodes"] if "Material" in node["type"]]

            if industry_nodes:
                input_text += f"  ├── **Industries:** {', '.join([node['name'] for node in industry_nodes])}\n"
            if cert_nodes:
                input_text += f"  ├── **Certifications:** {', '.join([node['name'] for node in cert_nodes])}\n"
            if manufactor_nodes:
                input_text += f"  ├── **Manufacturers:** {', '.join([node['name'] for node in manufactor_nodes])}\n"
            if material_nodes:
                input_text += f"  ├── **Materials:** {', '.join([node['name'] for node in material_nodes])}\n"

        # Show a Few-shot Example.
        input_text += """
                        ### Example:
                        User Question: Which manufacturers work in the material field and satisfy ISO9001?
                        Knowledge Context:
                        - Entity ID: 149401-us.all.biz, Name: 149401-us.all.biz
                        - Entity ID: 6939, Name: ISO9001 (Certification)
                        - Entity ID: 70, Name: Woods
                        Additional Related Information:
                        - 149401-us.all.biz processes Woods.
                        - 149401-us.all.biz is certified with ISO9001.
                        the Expected Answer should formulated as below:
                        (149401-us.all.biz)-[:Certification]-> (ISO9001).
                        (149401-us.all.biz)-[:Process]-> (Woods)

                        ### User Question:
                        """ + user_question + """

                        ### Answer:
                    """

        print("\n[LLM Prompt]\n" + input_text)
        return self.generate_with_llm(input_text)


    def generate_with_llm(self, input_text):
        """Generate response using Gemini LLM with output cleaning."""
        try:
            response = model.generate_content(input_text)
            return self.clean_response(response.text)
        except Exception as e:
            return "Sorry, I couldn't generate an answer due to an error."

    def clean_response(self, text):
        """Remove Markdown formatting and redundant symbols."""
        text = text.replace("*", "").replace("###", "").strip()
        return "\n".join([line.strip() for line in text.split("\n") if line.strip()])

    def rag_pipeline(self, user_question):
        """Execute the RAG pipeline with embedding-based retrieval and in-context learning."""
        start_time = time.time()

        relevant_entities = self.retrieve_relevant_entities(user_question)
        answer = self.generate_answer(user_question, relevant_entities)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[TIME] RAG pipeline execution time: {elapsed_time:.3f} seconds")

        return answer


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