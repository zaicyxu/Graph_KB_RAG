# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
@Project Name: KnowledgeGraph_ExpertSystem
@File Name: main_knoeledgebase_system.py
@Software: Python
@Time: Mar/2025
@Author: Yufei Quan, Rui Xu
@Version: 0.4.2
@Description: Knowledge graph-based expert system workflow.
              Compared with the RAG process, the preprocessing (embedding generation,
              keyword extraction, query depth calculation, and graph retrieval) remains the same.
              However, instead of using LLM, it employs rule-based analysis and template-based
              answer generation to serve as a control group for comparison.
"""


import pickle
import re
import numpy as np
from neo4j import GraphDatabase
from porlog import configuration
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import json


class Neo4jExpertSystem:
    def __init__(self, uri, user, password, cache_file="embedding_cache.pkl"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=60)
        self.cache_file = cache_file
        self.embedding_cache = self.load_cache()

    def close(self):
        self.driver.close()

    def load_cache(self):
        """Load embedding cache from a pickle file."""
        try:
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}

    def save_cache(self):
        """Save embedding cache to a pickle file."""
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
                text = record['name']
                embedding = self.get_embedding(text)
                if embedding:
                    update_query = """
                    MATCH (n) WHERE elementId(n) = $node_id 
                    SET n.embedding = $embedding
                    """
                    session.run(update_query, node_id=record['node_id'], embedding=embedding)
        print("Embeddings updated successfully.")

    def get_embedding(self, text):
        """
        Generate or fetch the embedding for a given text from cache.
        Note: This assumes an offline embedding generation function (or pre-trained model)
        to ensure consistency with the preprocessing steps in the RAG version.
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        try:
            # Placeholder: Replace with an actual offline embedding model
            embedding = np.random.rand(768).tolist()
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

    def analyze_query_intent_kb(self, query_text):
        """
        Analyze query intent using rule-based methods.
        Extract target entity types, relationship types, and filter conditions based on an enriched vocabulary.
        """
        query_text_lower = query_text.lower()

        # Enriched entity mapping dictionary with synonyms
        entity_mapping = {
            "manufacturer": "Manufactor",
            "producer": "Manufactor",
            "maker": "Manufactor",
            "company": "Manufactor",
            "firm": "Manufactor",
            "certification": "Certification",
            "certified": "Certification",
            "standard": "Certification",
            "industry": "Industry",
            "sector": "Industry",
            "business": "Business",
            "enterprise": "Business",
            "material": "Material",
            "substance": "Material",
            "raw": "Material"
        }

        # Enriched relationship mapping dictionary with synonyms
        relationship_mapping = {
            "certify": "Certify",
            "certified": "Certify",
            "accredit": "Certify",
            "belong": "Belong",
            "involve": "Belong",
            "process": "Process",
            "manufacture": "Process",
            "produce": "Process",
            "work on": "Work_on",
            "operate in": "Work_on",
            "engage in": "Work_on"
        }

        entities = []
        relationships = []
        filters = {}

        # Parse target entities based on the mapping dictionary
        for key, val in entity_mapping.items():
            if key in query_text_lower:
                entities.append(val)

        # Parse target relationships based on the mapping dictionary
        for key, val in relationship_mapping.items():
            if key in query_text_lower:
                relationships.append(val)

        iso_pattern = re.compile(r'\biso\d+\b', re.IGNORECASE)
        iso_matches = iso_pattern.findall(query_text)
        if iso_matches:
            # Use the first matching certification
            filters["Certification"] = iso_matches[0].upper()
        elif "certification" in query_text_lower:
            filters["Certification"] = "CERTIFICATION"

        # Determine question type based on common interrogative words
        if any(word in query_text_lower for word in ["does", "is", "do", "are"]):
            question_type = "yes_no"
        elif any(word in query_text_lower for word in ["what", "which", "list"]):
            question_type = "entity_lookup"
        else:
            question_type = "complex_relation"

        # Default entity if none detected
        if not entities:
            entities.append("Business")

        return {
            "entities": list(set(entities)),
            "relationships": list(set(relationships)),
            "filters": filters,
            "question_type": question_type
        }

    def retrieve_relevant_entities(self, query_text, top_k=5, similarity_threshold=0.7):
        """
        First tries to retrieve relevant entities using keyword-based Cypher graph search (with depth).
        If that fails or returns no results, fallback to embedding-based similarity retrieval (with depth).
        Returns enriched entity subgraphs for downstream reasoning.
        """

        depth = self.determine_query_depth(query_text)

        # Step 1: Keyword-based retrieval
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
        WHERE ({label_conditions}) AND any(kw IN $keywords WHERE toLower(n.Name) CONTAINS kw)
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

        # Step 2: Similarity fallback if keyword search fails
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            print("[WARNING] No embedding generated for query.")
            return []

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

        similarity_results = []
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
        Generate structured triples in [(subject, relation, object)] format.
        Returns empty list when no relevant entities found.
        """
        if not relevant_entities:
            return []

        triples = []

        for entity in relevant_entities:
            # Safely extract subject with fallback to empty string
            subject = entity.get("node1", {}).get("name", "").strip()
            if not subject:  # Skip invalid entities
                continue

            # Extract relations and connected nodes with type checking
            relations = entity.get("relations", [])
            connected_nodes = entity.get("connected_nodes", [])

            # Process each relation with index-based matching
            for idx, relation in enumerate(relations):
                # Clean relation name
                cleaned_relation = relation.strip() if relation else "UnknownRelation"

                # Get corresponding object with fallbacks
                obj_node = connected_nodes[idx] if idx < len(connected_nodes) else {}
                obj_name = obj_node.get("name", "Unknown").strip()

                triples.append((subject, cleaned_relation, obj_name))

        return triples[:5]

    def expert_pipeline(self, user_question):
        """
        Expert system workflow:
        """
        relevant_entities = self.retrieve_relevant_entities(user_question)
        return self.generate_answer(user_question, relevant_entities)

def main():
    expert_system = Neo4jExpertSystem(
        uri=configuration.NEO4J_URI,
        user=configuration.NEO4J_USER,
        password=configuration.NEO4J_PASSWORD
    )
    try:
        user_input = input("Enter your question: ")
        answer = expert_system.expert_pipeline(user_input)
        print("\nFinal Answer:\n", answer)
    finally:
        expert_system.close()

if __name__ == "__main__":
    main()
