import streamlit as st

from infrastructure.ui.message import write_message
from infrastructure.ui.session import get_session_id
from infrastructure.llm_openai.agent import SemanticSearchAgent
from infrastructure.neo4j_db.langchain_graph import graph

print(get_session_id())
prompt = """
You are a movie expert providing information about applications and technical description.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to software applications and their descriptions.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Forks
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

tool_name = "For when you need to search for information about apps based on their description or name"
tool_description = "For when you need to search for information about apps based on their description or name"

retriever_params = {}
retriever_params['index_name'] = "solution_description_vector"
retriever_params['node_label'] = "Solution"
retriever_params['text_node_property'] = "description"
retriever_params['embedding_node_property'] = "descriptionEmbedding"
retriever_params['retrieval_query'] = """
    RETURN
    node.description AS text,
    score,
    {
        name: node.name,
        platform: [ (node)-[:IMPLEMENTS]->(application) | application.platform ],
        useCases: [ (node)-[:ATTENDS]->(use) | [use.name, use.description] ],
        mainApis: [ (node)-[:IMPLEMENTS]->(application)-[:USE]->(api) | [api.resourceType, api.resourceName, api.description] ],
        connectedApis: [ (node)-[:IMPLEMENTS]->(application)-[:USE]->(api)-[:REQUEST]->(others) | [others.resourceType, others.resourceName, others.description] ],
        dataSources: [ (node)-[:IMPLEMENTS]->(application)-[:USE]->(api)-[:CONSUME]->(db) | [db.databaseName, db.resourceName, db.engine, db.resourceType] ],
        files: [ (node)-[:IMPLEMENTS]->(application)-[:USE]->(api)-[:READ]->(file) | [file.databaseName, file.resourceName, file.engine, file.resourceType] ]
    } AS metadata
"""

agent = SemanticSearchAgent(tool_name, tool_description, prompt, graph, retriever_params)


# Configuración de la página
st.set_page_config(page_title="Chatbot de aplicaciones", page_icon="🤖", layout="wide")

st.title("Chatbot de aplicaciones")
st.subheader("Este chatbot te ayudará a encontrar información sobre aplicaciones y sus descripciones técnicas.")

st.divider()
st.text("Aplicaciones:\n- Customer selector\n- Best Product Planner App\n- Smart pricing App\n- Discount estimator App")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, ¿Cómo puedo ayudarte?"},
    ]

def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Pensando...'):
        response = agent.generate_response(message, get_session_id)
        write_message('assistant', response)
        
# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("Qué información necesitas?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
