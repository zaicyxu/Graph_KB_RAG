# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: KnowledgeGraph_RAG
@File Name: query_generation.py
@Software: Python
@Time: Mar/2025
@Author: Rui Xu
@Contact: rxu@kth.se
@Version: 0.6.4
@Description: Automatically generates varied questions with spelling errors, synonyms,
              and different syntactic structures for testing similarity-based retrieval.
              Supports two types of questions.
"""

import random
import string
from neo4j import GraphDatabase
from porlog.configuration import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class QuestionGenerator:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        if self.driver:
            self.driver.close()

    def _get_nodes_by_label(self, label, limit=50):
        """
        Query the database for nodes whose Node_Type equals the given label.
        Returns a list of node names.
        """
        query = f"""
        MATCH (n:{label})
        RETURN DISTINCT n.Name AS name
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [record["name"] for record in result]

    def introduce_spelling_error(self, word, error_probability=0.1):
        """
        Introduce a spelling error into the given word with a certain probability.
        """
        if random.random() > error_probability or len(word) < 2:
            return word
        error_type = random.choice(['replace', 'swap'])
        word_list = list(word)
        if error_type == 'replace':
            index = random.randint(0, len(word_list) - 1)
            word_list[index] = random.choice(string.ascii_lowercase)
        elif error_type == 'swap' and len(word_list) > 1:
            index = random.randint(0, len(word_list) - 2)
            word_list[index], word_list[index + 1] = word_list[index + 1], word_list[index]
        return "".join(word_list)

    def replace_with_synonyms(self, word):
        """
        Replace given words with synonyms.
        """
        synonyms = {
            "manufacturer": "producer",
            "certification": "standard",
            "business": "enterprise",
            "material": "substance",
            "industry": "sector"
        }
        return synonyms.get(word.lower(), word)

    def process_text(self, text, apply_error=True):
        """
        Apply synonym replacement and optionally introduce spelling errors.
        """
        text = self.replace_with_synonyms(text)
        return self.introduce_spelling_error(text, error_probability=0.1) if apply_error else text

    def generate_questions(self, num_questions=20, apply_spelling_errors=True, question_type="category1"):
        """
        Generate questions based on simplified logic.
        """
        questions = []
        process_text = lambda text: self.process_text(text, apply_spelling_errors)

        if question_type == "category1":
            manufacturers = self._get_nodes_by_label("Manufactor", limit=50)
            businesses = self._get_nodes_by_label("Business", limit=50)
            industries = self._get_nodes_by_label("Industry", limit=50)

            if not manufacturers:
                print("Insufficient manufacturer nodes for category1 questions.")
                return []

            templates = []
            if businesses:
                for m in manufacturers:
                    for b in businesses:
                        m_proc = process_text(m)
                        b_proc = process_text(b)
                        templates.append([
                            f"How does {m_proc} establish and sustain its connection with {b_proc} through Sub_business relationship?",
                            "category1"
                        ])
                        templates.append([
                            f"In what way do manufacturers like {m_proc} interact with {b_proc} through Sub_business relationships?",
                            "category1"
                        ])

            # Industry
            if industries:
                for m in manufacturers:
                    for i in industries:
                        m_proc = process_text(m)
                        i_proc = process_text(i)
                        templates.append([
                            f"To what extent does {m_proc} maintain an active relationship with {i_proc} via Sub_industry relationships?",
                            "category1"
                        ])
                        templates.append([
                            f"What are the specific pathways through which {m_proc} maintains links to {i_proc} via Sub_industry?",
                            "category1"
                        ])

            if not templates:
                print("No templates generated for category1.")
                return []

            if len(templates) > num_questions:
                questions = random.sample(templates, num_questions)
            else:
                questions = templates.copy()
                while len(questions) < num_questions and templates:
                    questions.append(random.choice(templates))


        elif question_type == "category2":
            manufacturers = self._get_nodes_by_label("Manufactor", limit=50)
            certifications = self._get_nodes_by_label("Certification", limit=50)
            businesses = self._get_nodes_by_label("Business", limit=50)
            materials = self._get_nodes_by_label("Material", limit=50)
            industries = self._get_nodes_by_label("Industry", limit=50)

            if not manufacturers or not certifications:
                print("Insufficient nodes for category2 questions.")
                return []

            templates = []
            # Business templates:
            if businesses:
                for m in manufacturers:
                    for b in businesses:
                        for c in certifications:
                            m_proc = process_text(m)
                            b_proc = process_text(b)
                            c_proc = process_text(c)
                            templates.append([f"Does {m_proc} Work_on {b_proc} and comply with the {c_proc} same time?", "category2"])
                            templates.append(
                                [f"Which manufacturers Work_on {b_proc} and satisfy the {c_proc} at same time?", "category2"])
            # Material templates:
            if materials:
                for m in manufacturers:
                    for mat in materials:
                        for c in certifications:
                            m_proc = process_text(m)
                            mat_proc = process_text(mat)
                            c_proc = process_text(c)
                            templates.append([f"Does {m_proc} Process {mat_proc} and meet the {c_proc} at the same time?", "category2"])
                            templates.append(
                                [f"Which manufacturers Process {mat_proc} and satisfy the {c_proc} same time?", "category2"])
            # Industry templates:
            if industries:
                for m in manufacturers:
                    for ind in industries:
                        for c in certifications:
                            m_proc = process_text(m)
                            ind_proc = process_text(ind)
                            c_proc = process_text(c)
                            templates.append([f"Does {m_proc} Belong_to {ind_proc} and satisfy the {c_proc} same time?", "category2"])
                            templates.append(
                                [f"Which manufacturers Belong_to {ind_proc} and meet the {c_proc} same time?", "category2"])

            if not templates:
                print("No templates generated for category2.")
                return []
            # Sample or pad the list to obtain exactly num_questions questions
            if len(templates) > num_questions:
                questions = random.sample(templates, num_questions)
            else:
                questions = templates.copy()
                while len(questions) < num_questions and questions:
                    questions.append(random.choice(templates))

        random.shuffle(questions)
        return questions


if __name__ == "__main__":
    qg = QuestionGenerator()
    binary_questions = qg.generate_questions(num_questions=20, apply_spelling_errors=True, question_type="category1")
    # ternary_questions = qg.generate_questions(num_questions=20, apply_spelling_errors=True, question_type="category2")
    qg.close()

    print("Category1 Questions:")
    for q in binary_questions:
        print(q)
    # print("\nCategory2 Questions:")
    # for q in ternary_questions:
    #     print(q)

