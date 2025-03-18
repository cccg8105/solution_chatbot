from langchain_neo4j import Neo4jVector

from .langchain_graph import graph

class VectorRetriever:
    def __init__(self, embeddings, index_name, 
                 node_label, text_node_property, 
                 embedding_node_property, retrieval_query):
        self.neo4jvector = Neo4jVector.from_existing_index(
            embeddings,                              # (1)
            graph=graph,                             # (2)
            index_name=index_name,                 # (3)
            node_label=node_label,                      # (4)
            text_node_property=text_node_property,               # (5)
            embedding_node_property=embedding_node_property, # (6)
            retrieval_query=retrieval_query
        )
    
    def get_retriever(self):
        return self.neo4jvector.as_retriever()
