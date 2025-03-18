from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.tools import Tool

from infrastructure.llm_openai.document_chain import DocumentChain
from infrastructure.llm_openai.connector import llm, embeddings
from infrastructure.neo4j_db.vector_retriever import VectorRetriever

class SemanticSearchAgent:
    def __init__(self, tool_name, tool_description, 
                 agent_prompt_text, 
                 graph, 
                 retriever_params):
        self.graph = graph
        retriever = VectorRetriever(embeddings, 
                                    retriever_params['index_name'],
                                    retriever_params['node_label'], 
                                    retriever_params['text_node_property'],
                                    retriever_params['embedding_node_property'], 
                                    retriever_params['retrieval_query'])
        
        chain = DocumentChain(retriever.get_retriever(), llm)
        tools = [ 
            Tool.from_function(
                name = tool_name,  
                description = tool_description,
                func = chain.invoke_retriever, 
            )
        ]
        
        agent_prompt = PromptTemplate.from_template(agent_prompt_text)
        agent = create_react_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True
            )
        
        self.chat_agent = RunnableWithMessageHistory(
            agent_executor,
            self.get_memory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    
    def get_memory(self, session_id):
        return Neo4jChatMessageHistory(session_id=session_id, graph=self.graph)
    
    def generate_response(self, user_input, get_session_id):
        response = self.chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

        return response['output']
