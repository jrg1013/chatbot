# learn.py

import cfg

from langchain import document_loaders
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Local embedding model
embeddings_model_repo = SentenceTransformer('all-MiniLM-L6-v2')

loader = document_loaders.CSVLoader(
    file_path="./documents/Preguntas-Respuestas - ONLINE.csv",
    csv_args={
        "delimiter": ";",
        "quotechar": '"',
        "fieldnames": ["Intent", "Ejemplo mensaje usuario", "Respuesta"],
    })
documents = loader.load()

model_name = cfg.embeddings_model_repo
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=cfg.split_chunk_size,
                                      chunk_overlap=cfg.split_overlap)

texts = text_splitter.split_documents(documents)
len(texts)

vectordb = FAISS.from_documents(
    documents=texts,
    embedding=embeddings
)

vectordb.save_local(cfg.Embeddings_path)
