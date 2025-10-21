# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: KnowledgeGraph_RAG
@File Name: Verify_Query_Similarity.py
@Software: Python
@Time: Mar/2025
@Author: Yufei Quan, Rui Xu
@Contact: yufeiq@kth.se, rxu@kth.se
@Version: 0.5.0
@Description: Automatically generate questions and compare with Yes/No similarity calculate method.
"""

import csv
import importlib
from fuzzywuzzy import fuzz
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from query_generation import QuestionGenerator
from RAG_verify_query_similarity import Neo4jRAGSystem


class RetrievalEvaluator:
    def __init__(self, rag_system, question_generator,
                 output_file="query_evaluation_results_test.csv"):
        """
        Initialize the evaluator with a single RAG system instance (multi-method support).
        """
        self.rag_system = rag_system
        self.question_generator = question_generator
        self.output_file = output_file

    def evaluate_retrieval_methods(self, num_questions=10):
        """
        Evaluate retrieval methods by generating questions and obtaining the final answer via the RAG pipeline.
        """
        questions_with_metadata = self.question_generator.generate_questions(num_questions, apply_spelling_errors=True)

        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Question",
                "Ground_Truth",
                "Keyword_Final_Answer",
                "Similarity_Final_Answer",
                "Keyword_exact_accuracy",
                "Similarity_exact_accuracy",
                "Category"
            ])

            for question_entry in questions_with_metadata:
                question, category = question_entry
                print(f"\nProcessing question: {question} (Category: {category})")

                # Retrieve results using different methods
                keyword_results = self.rag_system.retrieve_relevant_entities(question,
                                top_k=3, use_similarity=False)
                similarity_results = self.rag_system.retrieve_relevant_entities(question,
                                    top_k=3, use_similarity=True)

                # Generate the final answers (only the final structured answer without the reasoning section)
                final_answer_keywords = self.rag_system.generate_answer(question, keyword_results)
                final_answer_similarity = self.rag_system.generate_answer(question, similarity_results)

                # Generate ground truth labels (set of strings)
                ground_truth = self._generate_ground_truth(question)

                # Calculate exact match accuracy by comparing final answer strings with ground truth
                keyword_exact_match_accuracy = self.compute_rouge_accuracy(final_answer_keywords, ground_truth)
                similarity_exact_match_accuracy = self.compute_rouge_accuracy(final_answer_similarity,
                                                                                       ground_truth)

                writer.writerow([
                    question,
                    "; ".join(ground_truth),
                    final_answer_keywords,
                    final_answer_similarity,
                    keyword_exact_match_accuracy,
                    similarity_exact_match_accuracy,
                    category
                ])

    def _generate_ground_truth(self, question):
        """
        Generate ground truth labels for a given question by querying the knowledge graph.
        The format is "Entity - Relationship -> Connected Entity".
        """
        entities = self.rag_system.extract_keywords(question)
        if not entities:
            print(f"[WARNING] No keywords extracted from `{question}`.")
            return set()

        ground_truth = set()
        with self.rag_system.driver.session() as session:
            for entity in entities:
                cypher_query = f"""
                MATCH (n)-[r]->(m)
                WHERE toLower(n.Name) CONTAINS toLower('{entity}')
                RETURN n.Name AS node_name, type(r) AS relation_name, m.Name AS connected_node
                LIMIT 20
                """
                result = session.run(cypher_query)
                for record in result:
                    ground_truth.add(f"{record['node_name']} - {record['relation_name']} -> {record['connected_node']}")
        if not ground_truth:
            print(f"[WARNING] No ground truth found for `{question}`.")
        return ground_truth

    def compute_rouge_accuracy(self, predicted_triples, ground_truth_triples, fuzzy_threshold=80):
        parsed_ground_truth = set()
        for triple in ground_truth_triples:
            match = re.match(r"^\s*([\w\-\.]+)\s*-\s*([\w\-]+)\s*->\s*([\w\-\.]+)\s*$", triple)
            if match:
                subj = match.group(1).strip().lower().replace("_", "")
                rel = match.group(2).strip().lower()
                obj = match.group(3).strip().lower().replace("_", "")
                parsed_ground_truth.add((subj, rel, obj))
            else:
                print(f"[WARNING] Invalid ground truth triple: {triple}")

        parsed_predicted = []
        for triple in predicted_triples:
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                subj = re.sub(r"[^\w\-\.]", "", str(triple[0])).strip().lower().replace("_", "")
                rel = re.sub(r"[^\w\-]", "", str(triple[1])).strip().lower()
                obj = re.sub(r"[^\w\-\.]", "", str(triple[2])).strip().lower().replace("_", "")
                parsed_predicted.append((subj, rel, obj))
            else:
                print(f"[WARNING] Invalid predicted triple format: {triple}")

        if not parsed_predicted:
            return 0.0

        match_count = 0
        for pred in parsed_predicted:
            if pred in parsed_ground_truth:
                match_count += 1
            else:
                best_score = max(
                    [(fuzz.ratio(pred[0], gt[0]) + fuzz.ratio(pred[1], gt[1]) + fuzz.ratio(pred[2], gt[2])) / 3
                     for gt in parsed_ground_truth]
                )
                if best_score >= fuzzy_threshold:
                    match_count += 1

        accuracy = match_count / len(parsed_predicted)
        return round(accuracy, 3)


    def _format_results(self, results):
        """
        Format search results for export to a CSV file.
        """
        if not results:
            return "No results"

        formatted_results = []
        for entity in results:
            node_name = entity["node1"]["name"]
            node_type = ", ".join(entity["node1"]["type"])
            relations = "; ".join(entity["relations"])
            connected_entities = ", ".join([c["name"] for c in entity["connected_nodes"]])
            formatted_results.append(
                f"{node_name} ({node_type}) | Relations: {relations} | Connected: {connected_entities}"
            )
        return " || ".join(formatted_results)


if __name__ == "__main__":
    # Import configuration dynamically
    config = importlib.import_module("configuration")

    # Initialize the question generator
    question_generator = QuestionGenerator()

    # Create a single RAG system instance (multi-method support)
    rag_system = Neo4jRAGSystem(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD
    )

    # Run evaluation (store all results in one CSV file)
    evaluator = RetrievalEvaluator(rag_system, question_generator)
    evaluator.evaluate_retrieval_methods(num_questions=20)

    # Close database connection
    rag_system.close()


