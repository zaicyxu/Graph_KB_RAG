# Install necessary libraries
# pip install -q -U google-generativeai

import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np

class GraphHandler:
    def __init__(self, csv_file):
        """
        Initialize the graph handler with the CSV data file.
        """
        self.csv_file = csv_file
        self.graph = nx.Graph()
        self.node_descriptions = {}
        self.node_embeddings = {}
        self.df = pd.read_csv(csv_file)
        self._build_graph()

    def _build_graph(self):
        """
        Build the graph from the CSV data. Nodes are stations, and edges
        are based on the 'Zone' column.
        """
        # Add nodes with attributes
        for _, row in self.df.iterrows():
            self.graph.add_node(
                row["Station"],
                os_x=row["OS X"],
                os_y=row["OS Y"],
                latitude=row["Latitude"],
                longitude=row["Longitude"],
                zone=row["Zone"],
                postcode=row["Postcode"],
                zone_original=row["Zone_original"],
            )

        # Add edges between stations in the same zone
        for zone in self.df["Zone"].unique():
            stations_in_zone = self.df[self.df["Zone"] == zone]["Station"].tolist()
            for i, station1 in enumerate(stations_in_zone):
                for station2 in stations_in_zone[i + 1:]:
                    self.graph.add_edge(station1, station2, zone=zone)

    def generate_node_descriptions(self):
        """
        Generate descriptions for each node (station) based on its attributes.
        """
        self.node_descriptions = {
            node: f"Station: {node}, Latitude: {data['latitude']}, Longitude: {data['longitude']}, Zone: {data['zone']}, Postcode: {data['postcode']}"
            for node, data in self.graph.nodes(data=True)
        }

    def generate_node_embeddings(self):
        """
        Generate embeddings for the node descriptions using a SentenceTransformer model.
        """
        embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.node_embeddings = {
            node: embedder.encode(description)
            for node, description in self.node_descriptions.items()
        }

    def find_most_relevant_node(self, query):
        """
        Find the most relevant node based on the query using cosine similarity.
        """
        embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        query_embedding = embedder.encode(query)
        max_similarity = -1
        most_relevant_node = None

        for node, embedding in self.node_embeddings.items():
            similarity = (query_embedding @ embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            if similarity > max_similarity:
                max_similarity = similarity
                most_relevant_node = node

        return most_relevant_node, max_similarity


class GeminiHandler:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        """
        Initialize the Gemini API handler with the API key and model.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt):
        """
        Generate a response using the Gemini API based on the provided prompt.
        """
        response = self.model.generate_content(prompt)

        # Print the response to inspect its structure
        print("Response structure:", response)

        # Try accessing the correct attribute
        if hasattr(response, '_result'):
            return response._result  # If _result exists, return it
        else:
            return "No result found in response."


def main():
    # Initialize GraphHandler with the correct CSV file path
    graph_handler = GraphHandler(csv_file=r"D:\Code\Gemini_RAG\london_transport_datasets_London_stations.csv")

    # Generate node descriptions and embeddings
    graph_handler.generate_node_descriptions()
    graph_handler.generate_node_embeddings()

    # Example query
    query = "Which important stations are located in Zone 3?"

    # Find the most relevant node for the query
    relevant_node, similarity = graph_handler.find_most_relevant_node(query)

    if relevant_node:
        context = graph_handler.node_descriptions[relevant_node]
        print(f"Most relevant station: {relevant_node} (Similarity: {similarity:.2f})")
        print(f"Context: {context}")

        # Initialize GeminiHandler with your API key
        gemini_handler = GeminiHandler(api_key="AIzaSyBdTPqt4RpQvOc676Z1v_OuEkDsqhrJd9k")

        # Generate a response using Gemini
        prompt = f"""
        My question is: {query}
        Based on the following station information, generate a response:
        {context}
        """
        response = gemini_handler.generate_response(prompt)
        print("Response generated by Gemini:")
        print(response)
    else:
        print("No relevant station found.")


if __name__ == "__main__":
    main()

