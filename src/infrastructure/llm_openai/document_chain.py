from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class DocumentChain:
    def __init__(self, vector_retriever, llm):
        instructions = (
            "Use the given context to answer the question."
            "If you don't know the answer, say you don't know."
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", instructions),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.chain_retriever = create_retrieval_chain(
            vector_retriever, 
            question_answer_chain
        )
    
    def invoke_retriever(self, input):
        return self.chain_retriever.invoke({"input": input})
    