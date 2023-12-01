# util.py

import cfg
import tokens

# vector stores
from langchain.vectorstores import FAISS

# models
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub

# Acceso al llm


def get_llm():
    huggingfacehub_api_token = tokens.huggingfacehub_api_token

    llm = HuggingFaceHub(
        repo_id=cfg.model_name,
        model_kwargs={
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "do_sample": cfg.do_sample,
            "num_return_sequences": cfg.num_return_sequences
        },
        huggingfacehub_api_token=huggingfacehub_api_token
    )

    return llm


# Local embedding model
model_name = cfg.embeddings_model_repo
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb = FAISS.load_local(
    cfg.Embeddings_path,
    embeddings
)
