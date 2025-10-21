# Graph_KB_RAG
For Leveraging the Graph-based LLM to Support the Analysis of Supply Chain Information coding project

## Details of coding file:
### 1. Query.sql
Used to construct graph datasets in Neo4j. and the original data file in info_MSKG, named Entity_Relation_of_MSKG.csv

### 2. query_generation.py
Used to automatically generate test questions.

### 3. main_RAG
1. configuration: Parameter configuration file.
2. embedding_cache.pkl: embedding information storage file.
3. main_rag_dynamatic_masked.py: Database partial mask test code.
4. main_rag_dynamatic_search.py: our main method.
5. main_rag_embedding.py: Pure embedding similarity retrieval method.
6. main_rag_muti-depth_search.py: Fixed-depth search method.

### 4. Verification_MSKG
Experimental code for verifying the superiority of different methods.
