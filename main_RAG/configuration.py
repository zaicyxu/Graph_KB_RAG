# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
@File Name: configuration.py
@Description: Store sensitive information like API keys and database credentials
"""

# Gemini API Key
GEMINI_API_KEY = 


# Neo4j Database Credentials
NEO4J_URI = 
NEO4J_BLOOM_URI = 
NEO4J_USER = 
NEO4J_PASSWORD = 

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
