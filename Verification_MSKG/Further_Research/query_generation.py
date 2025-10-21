# -*- coding: utf-8 -*-
# !/usr/bin/env python


"""
@Project Name: KnowledgeGraph_RAG
@File Name: query_generation.py
@Software: Python
@Time: Mar/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.2.3
@Description: Automatically generates varied questions with spelling errors, synonyms,
              and different syntactic structures for testing similarity-based retrieval.
"""


import random
import string
from neo4j import GraphDatabase
from porlog.configuration import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, RELATIONSHIP_MAPPING


class QuestionGenerator:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def fetch_graph_data(self):
        """
        Retrieve all node relationships from the Neo4j database.
        """
        query = """
        MATCH (n)-[r]->(m)
        RETURN DISTINCT 
            labels(n) AS node1_labels, n.Name AS node1_name, 
            type(r) AS relationship, 
            labels(m) AS node2_labels, m.Name AS node2_name
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [
                {
                    "node1_labels": record["node1_labels"],
                    "node1_name": record["node1_name"],
                    "relationship": record["relationship"],
                    "node2_labels": record["node2_labels"],
                    "node2_name": record["node2_name"]
                }
                for record in result
            ]

    def introduce_spelling_error(self, word, error_probability=0.1):
        """
        Introduce a spelling error in a word with a given probability.
        """
        if random.random() > error_probability or len(word) < 2:
            return word

        error_type = random.choice(['replace', 'swap', 'delete', 'insert'])
        word_list = list(word)

        if error_type == 'replace':
            index = random.randint(0, len(word_list) - 1)
            word_list[index] = random.choice(string.ascii_lowercase)
        elif error_type == 'swap' and len(word_list) > 1:
            index = random.randint(0, len(word_list) - 2)
            word_list[index], word_list[index + 1] = word_list[index + 1], word_list[index]
        elif error_type == 'delete':
            index = random.randint(0, len(word_list) - 1)
            del word_list[index]
        elif error_type == 'insert':
            index = random.randint(0, len(word_list))
            word_list.insert(index, random.choice(string.ascii_lowercase))

        return "".join(word_list)

    def replace_with_synonyms(self, word):
        """
        Replace words with their synonyms to enhance variation.
        """
        synonyms = {
            "manufacturer": "producer",
            "company": "firm",
            "certification": "standard",
            "material": "substance",
            "industry": "sector",
            "supplier": "vendor",
            "electric vehicle": "EV",
            "solar panel": "photovoltaic cell",
        }
        return synonyms.get(word.lower(), word)

    def generate_questions(self, num_questions=20, apply_spelling_errors=True):
        """
        Generate varied questions based on graph data.
        :param num_questions: Number of questions to generate.
        :param apply_spelling_errors: Whether to introduce spelling errors in entity names.
        :return: List of questions formatted as [question_text, category].
        """
        graph_data = self.fetch_graph_data()
        cat1 = []
        cat2 = []
        cat3 = []

        def process_text(text):
            text = self.replace_with_synonyms(text)
            return self.introduce_spelling_error(text) if apply_spelling_errors else text

        for data in graph_data:
            node1_raw = data["node1_name"]
            node2_raw = data["node2_name"]
            relationship = data["relationship"]
            expected_node2_label = RELATIONSHIP_MAPPING.get(relationship)

            node1 = process_text(node1_raw)
            node2 = process_text(node2_raw)

            # Category 1: Simple entity association
            if expected_node2_label:
                q1 = f"What {expected_node2_label}s are associated with {node1}?"
            else:
                q1 = f"What entities are associated with {node1}?"
            cat1.append([q1, "category1"])

            # Category 2: Relationship Matching
            if expected_node2_label:
                if relationship == "Certify":
                    q2 = f"Does {node1} meet the {expected_node2_label} standard?"
                elif relationship == "Belong":
                    q2 = f"Is {node1} operating in the {expected_node2_label} industry?"
                elif relationship == "Process":
                    q2 = f"Does {node1} process {expected_node2_label}?"
                elif relationship == "Work_on":
                    q2 = f"Does {node1} work on {expected_node2_label}?"
                else:
                    q2 = f"Does {node1} have a {relationship} relationship with {node2}?"
            else:
                q2 = f"Does {node1} have a {relationship} relationship with {node2}?"
            cat2.append([q2, "category2"])

            # Category 3: Multi-layer Relationship Nesting
            if len(graph_data) > 1:
                data2 = random.choice(graph_data)
                relationship2 = data2["relationship"]
                expected_node2_label_2 = RELATIONSHIP_MAPPING.get(relationship2)
                node3 = process_text(expected_node2_label_2) if expected_node2_label_2 else process_text(
                    data2["node2_name"])
                q3 = f"Which {expected_node2_label if expected_node2_label else 'entity'} associated with {node1} also has a {relationship2} relationship with {node3}?"
            else:
                q3 = q2
            cat3.append([q3, "category3"])

        num_per_cat = num_questions // 3
        questions = []
        questions.extend(random.sample(cat1, min(num_per_cat, len(cat1))))
        questions.extend(random.sample(cat2, min(num_per_cat, len(cat2))))
        questions.extend(random.sample(cat3, min(num_per_cat, len(cat3))))

        while len(questions) < num_questions:
            questions.append(random.choice(cat1 + cat2 + cat3))

        random.shuffle(questions)
        return questions[:num_questions]


if __name__ == "__main__":
    question_gen = QuestionGenerator()
    questions = question_gen.generate_questions(num_questions=50, apply_spelling_errors=True)
    question_gen.close()

    for q in questions:
        print(q)
