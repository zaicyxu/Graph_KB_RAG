# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: KnowledgeGraph_RAG
@File Name: main_rag_pipeline.py
@Software: Python
@Time: Feb/2025
@Author: Yufei Quan, Rui Xu
@Contact: yufeiq@kth.se, rxu@kth.se
@Version: 7.2
@Description: Node-based RAG pipeline with Gemini embeddings and pickle-based caching.
"""

import pickle
import numpy as np
from neo4j import GraphDatabase
import google.generativeai as genai
from porlog import configuration

# Configure Gemini API
genai.configure(api_key=configuration.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


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

    def update_embeddings(self):
        """Update missing embeddings for documents in the Neo4j database."""
        with self.driver.session() as session:
            cypher_query = """
            MATCH (n)
            WHERE (n:Question OR n:Answer) AND n.embedding IS NULL
            RETURN n.title AS title, n.body_markdown AS body, elementId(n) AS node_id
            """
            records = session.run(cypher_query)

            for record in records:
                text = record['title'] + " " + record['body']
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

    def retrieve_relevant_docs(self, query_text, top_k=3, similarity_threshold=0.3):  # 修改点2: 降低阈值
        """Retrieve top-k relevant documents using node-based embedding matching with dot product similarity."""
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            print("No embedding generated for query text.")
            return []
        print("Query Embedding:", query_embedding)
        query_embedding = np.array(query_embedding)

        with self.driver.session() as session:
            cypher_query = """
            MATCH (n)
            WHERE (n:Question OR n:Answer) AND n.embedding IS NOT NULL
            RETURN n.title AS title, n.body_markdown AS body, n.embedding AS embedding
            """
            records = session.run(cypher_query)

            docs_with_scores = []
            for record in records:
                doc_embedding = np.array(record['embedding'])
                similarity = np.dot(query_embedding, doc_embedding)  # Dot product similarity
                if similarity >= similarity_threshold:
                    docs_with_scores.append({
                        'title': record['title'],
                        'body': record['body'][:500] + "..." if len(record['body']) > 500 else record['body'],
                        'score': similarity
                    })

            docs_with_scores.sort(key=lambda x: x['score'], reverse=True)
            return docs_with_scores[:top_k]

    def generate_answer(self, user_question, relevant_docs):
        """Generate an answer using Gemini LLM with optimized in-context prompt."""

        input_text = """You are an expert AI assistant specializing in programming, databases, and Neo4j.
        Your task is to answer technical questions based on the provided relevant documents.
        If the documents contain partial but relevant knowledge, infer the missing details based on logical reasoning.

        ### Knowledge Context:
        Below are retrieved documents that may contain useful information:

        """
        for doc in relevant_docs:
            input_text += f"Title: {doc['title']}\nContent: {doc['body']}\n\n"

        # Few-shot Example
        input_text += """
        ### Example:
        User Question: How to create an index in Neo4j?
        Knowledge Context:
        Title: Neo4j Indexing Guide
        Content: In Neo4j, you can create an index using `CREATE INDEX FOR (n:Label) ON (n.property)`.
        ---
        Final Answer:
        To create an index in Neo4j, use:
        ```cypher
        CREATE INDEX FOR (n:Label) ON (n.property);
        ```
        This will speed up queries on `n.property`.

        ### User Question:
        """ + user_question + """

        input_text += """

        ### Answer:

        return self.generate_with_llm(input_text)

    def generate_with_llm(self, input_text):
        """
        Generate
        response
        using
        Gemini
        LLM
        with output cleaning.
        """
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
        """Execute the RAG pipeline with node-based retrieval and in -context learning."""
        relevant_docs = self.retrieve_relevant_docs(user_question)
        return self.generate_answer(user_question, relevant_docs)


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
