# -*- coding: utf-8 -*-
import time
from py2neo import Graph
import google.generativeai as genai
from porlog.configuration import GEMINI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# 初始化 Gemini
genai.configure(api_key=GEMINI_API_KEY)

# 初始化 Neo4j 图连接
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 缓存
embedding_cache = {}

def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        embedding = response["embedding"]
        embedding_cache[text] = embedding
        return embedding
    except Exception as e:
        print(f"❌ Error generating embedding for '{text}': {e}")
        return None

def update_all_embeddings():
    target_types = ['manufacturer', 'certification', 'industry', 'service', 'material', 'energy']


    query = """
    MATCH (n)
    WHERE n.Node_Type IN $types
    RETURN n.Name AS name, elementId(n) AS node_id, n.Node_Type AS type
    """
    nodes = graph.run(query, types=target_types).data()

    print(f"🔍 Found {len(nodes)} target nodes to embed...")

    for idx, node in enumerate(nodes, 1):
        name = node["name"]
        node_id = node["node_id"]
        node_type = node["type"]

        embedding = get_embedding(name)
        if embedding is None:
            print(f"⚠️ Skipping {name}")
            continue

        # 显式转为浮点数列表
        embedding = list(map(float, embedding))

        graph.evaluate("""
            MATCH (n) WHERE elementId(n) = $node_id
            SET n.embedding = $embedding
            RETURN n
        """, node_id=node_id, embedding=embedding)

        print(f"✅ [{idx}/{len(nodes)}] Embedded '{name}' ({node_type})")
        time.sleep(0.05)

    print("🎉 All embeddings updated.")

if __name__ == "__main__":
    update_all_embeddings()
