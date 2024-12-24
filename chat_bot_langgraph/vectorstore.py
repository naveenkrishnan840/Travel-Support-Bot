from langchain.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os


loader = WebBaseLoader("https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md").load()


splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
splitter_docs = splitter.split_documents(loader)

os.environ["GOOGLE_API_KEY"]="AIzaSyAYmSmi6qAlKEALP0FtjK60ZsjMIjZv4z4"


client_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory="chat_bot_langgraph\chroma_db_folder",
    anonymized_telemetry=False,
)

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# vectorstore = Chroma.from_documents(documents=splitter_docs,
#     collection_name="travel_swiss_faq", embedding=gemini_embeddings
# )

from langchain.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever

vectorstore = FAISS.from_documents(documents=splitter_docs, embedding=gemini_embeddings)

vectorstore.save_local(folder_path="chat_bot_langgraph/database/faiss_db",
                       index_name="faiss_db_index")

print(vectorstore)

