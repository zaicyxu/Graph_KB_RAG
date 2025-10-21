# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
@Project Name: KnowledgeGraph_ExpertSystem
@File Name: Verify_semantic_analysis.py
@Software: Python
@Time: Mar/2025
@Author: Yufei Quan, Rui Xu
@Version: 0.1.2
@Description: Comparison of the impact of semantic analysis on answer accuracy
"""

import csv
import re
from main_RAG.main_rag_dynamatic_search import Neo4jRAGSystem
from rag_categoary_depth import Neo4jRAGSystem
from query_generation import QuestionGenerator
from porlog import configuration
from main_RAG import configuration_2


class ExperimentRunner:
    def __init__(self, top_k=3, similarity_threshold=0.7, num_questions=50):
        """
        Initialize the experiment runner: instantiate three RAG systems and the question generator.
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.num_questions = num_questions

        # Initialize the three RAG systems.
        self.rag_dynamic = Neo4jRAGSystem(
            configuration.NEO4J_URI, configuration.NEO4J_USER, configuration.NEO4J_PASSWORD
        )
        self.rag_category = Neo4jRAGSystem(
            configuration_2.NEO4J_URI, configuration.NEO4J_USER, configuration.NEO4J_PASSWORD
        )
        # Initialize the question generator.
        self.question_generator = QuestionGenerator()

    def extract_labels_from_answer(self, answer_text):
        """
        Extract labels from the structured answer string using regex pattern matching.
        """
        if not isinstance(answer_text, str):
            answer_text = str(answer_text)

        pattern = r"\((.*?)\)-\[:(.*?)\]->\((.*?)\)"
        matches = re.findall(pattern, answer_text)
        labels = set()
        for match in matches:
            labels.update(match)
        return list(labels)

    def compute_rouge_accuracy(self, predicted_triples, ground_truth_triples, fuzzy_threshold=80):
        """
        Compute structured accuracy by comparing predicted structured triples with ground truth triples using fuzzy matching.

        """
        import re
        from fuzzywuzzy import fuzz

        def normalize_triple(triple):
            # If triple is a string in the format "(sub)-[:rel]->(obj)", parse it.
            if isinstance(triple, str):
                m = re.match(r"\(([^)]+)\)-\[:([^]]+)\]->\(([^)]+)\)", triple)
                if m:
                    subj, rel, obj = m.group(1), m.group(2), m.group(3)
                else:
                    subj, rel, obj = triple, "", ""
            else:
                subj, rel, obj = triple[0], triple[1], triple[2]
            # Remove non-alphanumeric characters and lower-case
            subj_norm = re.sub(r'[^a-zA-Z0-9]', '', subj).lower().strip()
            rel_norm = re.sub(r'[^a-zA-Z0-9]', '', rel).lower().strip()
            obj_norm = re.sub(r'[^a-zA-Z0-9]', '', obj).lower().strip()
            return (subj_norm, rel_norm, obj_norm)

        # Normalize predicted triples
        normalized_pred = [normalize_triple(triple) for triple in predicted_triples]
        # Normalize ground truth triples (if they are strings, we'll convert them)
        normalized_gt = [normalize_triple(triple) for triple in ground_truth_triples]

        if not normalized_pred or not normalized_gt:
            return 0.0

        match_count = 0
        for pred in normalized_pred:
            best_score = 0.0
            # Compare each predicted triple with all ground truth triples
            for gt in normalized_gt:
                subj_score = fuzz.ratio(pred[0], gt[0])
                rel_score = fuzz.ratio(pred[1], gt[1])
                obj_score = fuzz.ratio(pred[2], gt[2])
                avg_score = (subj_score + rel_score + obj_score) / 3.0
                best_score = max(best_score, avg_score)
            if best_score >= fuzzy_threshold:
                match_count += 1

        accuracy = match_count / len(normalized_pred)
        return round(accuracy, 3)

    def _generate_ground_truth(self, question):
        """
        Generate ground truth labels for a given question by querying the knowledge graph.
        The format is "Entity - Relationship -> Connected Entity".
        """
        entities = self.rag_dynamic.extract_keywords(question)
        if not entities:
            print(f"[WARNING] No keywords extracted from `{question}`.")
            return set()

        ground_truth = set()
        with self.rag_dynamic.driver.session() as session:
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

    def run_experiment(self, output_file="experiment_results_1.csv"):
        """
        Run the experiment:
          1. Generate questions (each in the format [question_text, category]) using the QuestionGenerator.
          2. For each question, call the Category RAG system (which now expects a string, not a list).
          3. Generate ground truth from the knowledge graph.
          4. Compute the ROUGE-based exact accuracy.
          5. Write results horizontally (one row per test) in the CSV file.
        """
        questions_with_categories = self.question_generator.generate_questions(self.num_questions)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            header = [
                "Question",
                "Ground Truth",
                "Category Answer",
                "Category Exact Accuracy",
                "Category"
            ]
            writer.writerow(header)

            for question_entry in questions_with_categories:
                # Each question entry should be in the format [question_text, category]
                if isinstance(question_entry, list) and len(question_entry) >= 2:
                    actual_question = question_entry[0]
                    category = question_entry[-1]
                else:
                    print(f"[WARNING] Invalid question entry: {question_entry}")
                    continue

                print(f"\nProcessing question: {actual_question} (Category: {category})")

                # Concatenate the actual question and category into a single string for rag_pipeline
                category_query = actual_question + " " + category
                category_answer = self.rag_category.rag_pipeline(category_query)

                ground_truth = self._generate_ground_truth(actual_question)
                category_labels = self.extract_labels_from_answer(category_answer)

                exact_accuracy_category = self.compute_rouge_accuracy(category_labels, ground_truth)

                writer.writerow([
                    actual_question,
                    "\n".join(ground_truth),
                    category_answer,
                    exact_accuracy_category,
                    category
                ])

                print(f"Question: {actual_question}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Category Answer: {category_answer}")
                print(f"Category Exact Accuracy: {exact_accuracy_category}")
                print(f"Category: {category}")

    def close(self):
        """
        Close all connections and clean up resources.
        """
        self.question_generator.close()
        self.rag_dynamic.close()
        self.rag_category.close()

if __name__ == "__main__":
    experiment = ExperimentRunner(top_k=5, similarity_threshold=0.7, num_questions=50)
    experiment.run_experiment("experiment_results_1.csv")
    experiment.close()
