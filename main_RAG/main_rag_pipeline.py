# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
@Project Name: KnowledgeGraph_RAG
@File Name: main_rag_pipeline.py
@Software: Python
@Time: Feb/2025
@Author: Yufei Quan, Rui Xu
@Contact: yufeiq@kth.se, rxu@kth.se
@Version: 6.1
@Description: Enhanced RAG pipeline with Gemini embeddings and Neo4j-based vector search.
"""

from neo4j import GraphDatabase
import google.generativeai as genai
from porlog import configuration
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini API
genai.configure(api_key=configuration.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

class Neo4jRAGSystem:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_embedding(self, text):
        """Generate embedding using Gemini API."""
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def update_embeddings(self):
        """Update embeddings for all Question nodes using Gemini."""
        query = "MATCH (q:Question) RETURN q.title AS title"
        with self.driver.session() as session:
            results = session.run(query).data()

        for record in results:
            title = record["title"]
            embedding = self.get_embedding(title)

            if embedding is not None:
                update_query = """
                MATCH (q:Question {title: $title})
                SET q.embedding = $embedding
                """
                with self.driver.session() as session:
                    session.run(update_query, title=title, embedding=embedding)

    def generate_with_llm(self, input_text):
        """Generate answer using Gemini LLM."""
        try:
            response = model.generate_content(input_text)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating response from Gemini: {e}")
            return "Sorry, I couldn't generate an answer due to an error."

    def retrieve_relevant_docs(self, query_text, top_k=5):
        """Retrieve top-k relevant documents with Gemini embeddings."""
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            return []

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
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                docs_with_scores.append({
                    'title': record['title'],
                    'body': record['body'],
                    'score': similarity
                })

            docs_with_scores.sort(key=lambda x: x['score'], reverse=True)
            return docs_with_scores[:top_k]

    def generate_answer(self, user_question, relevant_docs):
        input_text = """You are an AI assistant for Neo4j. Answer based on these documents:
        ### Documents:
        """
        for doc in relevant_docs:
            input_text += f"Title: {doc['title']}\nContent: {doc['body']}\n\n"
        input_text += f"### Question: {user_question}\nAnswer:"
        return self.generate_with_llm(input_text)

    def rag_pipeline(self, user_question):
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
        print("\nAnswer:", answer)
    finally:
        rag_system.close()

if __name__ == "__main__":
    main()