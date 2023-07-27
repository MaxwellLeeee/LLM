import logging
import os

from langchain.chains import ConversationalRetrievalChain
from tqdm import tqdm
from langchain.vectorstores import FAISS
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

embeddings = HuggingFaceEmbeddings()
root_dir = "./UnrealEngine-5.1.1-release/Engine"
docs = []
for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

logging.info("start split")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(docs)
vectordb = FAISS.from_documents(texts, embeddings)
vectordb.save_local("faiss_index_ios")
logging.info("finish")
