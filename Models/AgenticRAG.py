from NormalRag import NormalRAG
import os
import json
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage

class AgenticRAG:
    """
    An Agentic RAG system that uses a Langchain agent to orchestrate RAG operations
    and potentially other tools.
    """
    def __init__(self, llm_name="openai", api_key=None, model_name=None, base_url=None):
        self.rag_core = Normal_RAG(llm_name=llm_name, api_key=api_key, model_name=model_name, base_url=base_url)
        self.llm_agent = self.rag_core.llm # The same LLM instance for the agent's reasoning

        # Define tools from the Normal_RAG functionalities
        # We use @tool decorator to expose these methods as Langchain tools
        @tool
        def retrieve_documents(query: str, top_k: int = 4) -> str:
            """
            Useful for retrieving relevant document chunks based on a natural language query.
            Input should be a clear, concise question. Returns a concatenated string of context.
            """
            context = self.rag_core.retrieve_context(query, top_k)
            if not context:
                return "No relevant documents found."
            return "\n\n".join(context)

        @tool
        def generate_final_answer(question: str, context: str) -> str:
            """
            Useful for generating a final, concise answer once relevant context has been retrieved.
            Requires both the original question and the retrieved context as input.
            """
            context_list = context.split("\n\n") # Re-split if retrieve_documents joins them
            return self.rag_core.generate_answer_from_context(question, context_list)

        self.tools = [retrieve_documents, generate_final_answer]

        # Define the agent's prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are an expert assistant designed to answer user questions by intelligently using the provided tools. "
                                      "First, use the 'retrieve_documents' tool to get relevant context. "
                                      "Then, use the 'generate_final_answer' tool with the retrieved context and the original question to provide a concise answer."),
                HumanMessage(content="{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create the agent
        self.agent = create_tool_calling_agent(self.llm_agent, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)


    def process_query(self, query: str) -> str:
        """
        Processes a user query using the agentic workflow.
        The agent will decide which tools to use to answer the question.
        """
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"An error occurred during agent processing: {e}"

    # Expose core RAG functionalities for direct management (e.g., loading/saving)
    def load_documents(self, dataset_folder):
        self.rag_core.load_documents(dataset_folder)

    def save_vector_store(self, index_name="faiss_index"):
        self.rag_core.save_vector_store(index_name)

    def load_vector_store(self, index_name="faiss_index"):
        self.rag_core.load_vector_store(index_name)