# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
@File Name: configuration.py
@Description: Store sensitive information like API keys and database credentials
"""

# Gemini API Key
GEMINI_API_KEY = "AIzaSyBdTPqt4RpQvOc676Z1v_OuEkDsqhrJd9k"
# GEMINI_API_KEY = "AIzaSyCg8PHoIJhvIukiooFg6b7K2Bi8r-EkhEQ"
# GEMINI_API_KEY = "AIzaSyC2LaeLTBOj-qISv_OxdKJPUQzwEbJ595Q"

# Neo4j Database Credentials
NEO4J_URI = "bolt://localhost:7687"
# NEO4J_BLOOM_URI = "neo4j://localhost:7687"
NEO4J_BLOOM_URI = "http://localhost:7474/bloom"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# Key entity types and relationships
KEY_ENTITIES = {
    "Manufacturer": ["Work_on", "Certify"],
    "Product": ["Process"],
    "Industry": ["Belong", "Sub_Industry"],
    "Certification": ["Certify"]
}

RELATIONSHIP_MAPPING = {
    "Belong": "Industry",
    "Certify": "Certification",
    "Process": "Material",
    "Work_on": "Service",
    "Sub_Business": "Service",
    "Sub_Industry": "Industry"
}