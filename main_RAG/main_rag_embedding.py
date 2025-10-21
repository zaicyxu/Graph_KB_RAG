# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: KnowledgeGraph_RAG
@File Name: main_rag_pipeline.py
@Software: Python
@Time: Feb/2025
@Author: Yufei Quan, Rui Xu
@Contact: yufeiq@kth.se, rxu@kth.se
@Version: 0.8.2
@Description: Node-based RAG pipeline with Gemini embeddings and pickle-based caching, and use embedding for matching.
"""

import pickle
import numpy as np
from neo4j import GraphDatabase
import google.generativeai as genai
from main_RAG import configuration
import time

# Configure Gemini API
genai.configure(api_key=configuration.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


class Neo4jRAGSystemNonedepth:
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
        """Compute the cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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

    def retrieve_related_graph_embeddings(self, node_id):
        """
        Retrieve the embeddings for all nodes and relationships connected to a given node.

        :param node_id: The ID of the node to retrieve the related graph for.
        :return: A dictionary containing embeddings for nodes and relationships.
        """
        with self.driver.session() as session:
            # Cypher search，getting the nodes and edges by the given node.
            cypher_query = """
            MATCH (n)-[r:Belong|Certify|Process|Sub_Bussiness|Sub_Industry|Work_on]->(m)
            WHERE n.Id = $node_id AND r.embedding IS NOT NULL
            RETURN n, r, m
            """
            records = session.run(cypher_query, node_id=node_id)

            graph_embeddings = {
                'nodes': [],
                'relationships': []
            }

            for record in records:
                # embedding node
                if record['n'].get('embedding'):
                    graph_embeddings['nodes'].append({
                        'id': record['n'].id,
                        'embedding': record['n'].get('embedding')
                    })
                # embedding edge
                if record['r'].get('embedding'):
                    graph_embeddings['relationships'].append({
                        'id': record['r'].id,
                        'embedding': record['r'].get('embedding')
                    })
                # embedding item node
                if record['m'].get('embedding'):
                    graph_embeddings['nodes'].append({
                        'id': record['m'].id,
                        'embedding': record['m'].get('embedding')
                    })

            return graph_embeddings

    def retrieve_relevant_entities(self, query_text, top_k=3, similarity_threshold=0.4, use_graph_embeddings=False):
        """Retrieve top-k relevant entities, optionally including their related graph embeddings."""
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            print("No embedding generated for query text.")
            return []

        query_embedding = np.array(query_embedding)

        with self.driver.session() as session:
            cypher_query = """
            MATCH (n)
            WHERE n.Node_Type IN ['Business', 'Certification', 'Industry', 'Manufactor', 'Material']
              AND n.embedding IS NOT NULL
            RETURN n.Id AS id, n.Name AS name, n.Node_Type AS type, n.embedding AS embedding
            """
            records = session.run(cypher_query)

            entities_with_scores = []
            for record in records:
                entity_embedding = np.array(record['embedding'])
                similarity = self.cosine_similarity(query_embedding, entity_embedding)
                if similarity >= similarity_threshold:
                    entities_with_scores.append({
                        'id': record['id'],
                        'name': record['name'],
                        'type': record['type'],
                        'score': similarity
                    })

            # **新增**：打印所有匹配到的实体及相似度
            print("\n[Retrieved Entities] Matching entities based on similarity:")
            for entity in entities_with_scores:
                print(
                    f"ID: {entity['id']} | Name: {entity['name']} | Type: {entity['type']} | Similarity: {entity['score']:.4f}")

            entities_with_scores.sort(key=lambda x: x['score'], reverse=True)

            if use_graph_embeddings:
                for entity in entities_with_scores:
                    graph_embeddings = self.retrieve_related_graph_embeddings(entity['id'])
                    entity['graph_embeddings'] = graph_embeddings

            return entities_with_scores[:top_k]

    def retrieve_related_graph_embeddings(self, node_id):
        """
        Retrieve the embeddings for all nodes and relationships connected to a given node.
        """
        with self.driver.session() as session:
            cypher_query = """
            MATCH (n)-[r:Belong|Certify|Process|Sub_Bussiness|Sub_Industry|Work_on]->(m)
            WHERE n.Id = $node_id
            RETURN n.Id AS source_id, n.Name AS source_name, m.Id AS target_id, m.Name AS target_name, r.type AS relationship
            """
            records = session.run(cypher_query, node_id=node_id)

            graph_data = {'nodes': set(), 'relationships': []}

            for record in records:
                graph_data['nodes'].add((record['source_id'], record['source_name']))
                graph_data['nodes'].add((record['target_id'], record['target_name']))
                graph_data['relationships'].append({
                    'source': record['source_id'],
                    'target': record['target_id'],
                    'relationship': record['relationship']
                })

            # **新增**：打印子图信息
            print(f"\n[Subgraph for Node {node_id}] Retrieved relationships:")
            for edge in graph_data['relationships']:
                print(f"{edge['source']} -[{edge['relationship']}]-> {edge['target']}")

            graph_data['nodes'] = [{"id": nid, "name": name} for nid, name in graph_data['nodes']]
            return graph_data

    def generate_answer(self, user_question, relevant_entities):
        """
        Generate an answer using Gemini LLM with optimized in-context prompt.
        """
        print("\n[Final Entities Used for Generation] Entities sent to LLM:")
        for entity in relevant_entities:
            print(f"ID: {entity['id']} | Name: {entity['name']} | Type: {entity.get('type', 'Unknown')}")

        input_text = """You are an expert AI assistant specializing in knowledge graph analysis and inference.
    Your task is to answer technical questions based on the provided knowledge context.
    Even if some details are missing, you should infer and provide a reasonable answer using domain expertise.

    ### Knowledge Context:
    Below are the retrieved entities from the knowledge graph along with their relationships:
    """
        for entity in relevant_entities:
            input_text += f"- Entity ID: {entity['id']}, Name: {entity['name']}, Type: {entity.get('type', 'Unknown')}\n"

        input_text += """
    ### Example:
    User Question: Which manufacturers work in the material field and satisfy ISO9001?
    Knowledge Context:
    - Manufacturer A is connected to Material X through Process.
    - Manufacturer A is certified with ISO9001 through Certify.
    Final Answer:
    Manufacturer A works in the material field and is ISO9001 certified.

    ### User Question:
    """ + user_question + """

    ### Answer:
    """
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
    rag_system = Neo4jRAGSystemNonedepth(
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