import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser, Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_groq import ChatGroq

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# # Set LangSmith environment variables
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"
# os.environ["LANGCHAIN_PROJECT"] = "interview"

# logger.debug(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
# logger.debug(f"LANGCHAIN_ENDPOINT: {os.getenv('LANGCHAIN_ENDPOINT')}")
# logger.debug(f"LANGCHAIN_API_KEY: {os.getenv('LANGCHAIN_API_KEY')}")
# logger.debug(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")

class RAGPipeline:
    def __init__(self, folder_path: str, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.folder_path = folder_path
        self.embedding_model_name = embedding_model_name
        self.chat_history = []
        self.llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key="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")
        self.store = LocalFileStore("./cache/")
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.embedding_function = CacheBackedEmbeddings.from_bytes_store(
            self.embedding_function, self.store, namespace=self.embedding_function.model_name
        )
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.setup_pipeline()

    def load_documents(self) -> List[Document]:
        documents = []
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            else:
                logger.warning(f"Unsupported file type: {filename}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)

    def create_vector_store(self, splits: List[Document]):
        self.vectorstore = Chroma.from_documents(
            collection_name="my_collection", documents=splits,
            embedding=self.embedding_function, persist_directory="./chroma_db"
        )

    def initialize_retriever(self):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

    def create_conversational_retrieval_chain(self):
        #with tracing_v2_enabled(project_name="interview"):
            contextualize_q_system_prompt = """
            Given a chat history and the latest user question
            which might reference context in the chat history,
            formulate a standalone question which can be understood
            without the chat history. Do NOT answer the question,
            just reformulate it if needed and otherwise return it as is.
            """

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, contextualize_q_prompt
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant designed to assist AI/ML engineers in interview preparation. 
                For every question, provide two types of answers:
                1. A **short answer**: A concise and direct response to the question.
                2. A **long answer**: A detailed explanation with examples, use cases, or technical depth.
                 
                 Example:
            User Question: What is overfitting in machine learning?
            Short Answer: Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on unseen data.
            Long Answer: Overfitting happens when a machine learning model captures noise or random fluctuations in the training data as if they were significant patterns. This results in a model that performs exceptionally well on the training data but fails to generalize to new, unseen data. Techniques to prevent overfitting include cross-validation, regularization (e.g., L1/L2 regularization), and using more training data.

            Follow this format for every question.
                """),
                ("system", "Context: {context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            self.qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def setup_pipeline(self):
        documents = self.load_documents()
        logger.info(f"Loaded {len(documents)} documents.")

        splits = self.split_documents(documents)
        logger.info(f"Split into {len(splits)} chunks.")

        self.create_vector_store(splits)
        logger.info("Vector store created.")

        self.initialize_retriever()
        logger.info("Retriever initialized.")

        self.create_conversational_retrieval_chain()
        logger.info("Conversational retrieval chain created.")

    def ask_question(self, question: str):
        response = self.qa_chain.invoke({"input": question, "chat_history": self.chat_history})
        self.chat_history.extend({"human": question, "ai": response["answer"]})
        return response["answer"]

if __name__ == "__main__":
    folder_path = r"C:\Users\dhira\Desktop\AI_Interview_RAG\data"
    rag_pipeline = RAGPipeline(folder_path)
    response = rag_pipeline.ask_question("how we can evaluate?")
    print(response)
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
   


