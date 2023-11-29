from langchain import document_loaders
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Local embedding model
embeddings_model_repo = SentenceTransformer('all-MiniLM-L6-v2')

loader = document_loaders.TextLoader(
    "./documents/Listado Preguntas-Respuestas - ONLINE.txt")
documents = loader.load()

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)

texts = text_splitter.split_documents(documents)
len(texts)

vectordb = FAISS.from_documents(
    documents=texts,
    embedding=embeddings
)

vectordb.save_local("faiss_index_hp")
