# util.py

import cfg
import tokens

# vector stores
from langchain.vectorstores import FAISS

# models
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub

# prompts
from langchain.prompts import PromptTemplate

# retrievers
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

# memory
from langchain.memory import ConversationBufferMemory

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

# Import vector database


def import_vectordb():

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

    return embeddings, vectordb

# Get Prompt based in the context and template in config


def get_prompt():
    template = cfg.template
    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])
    return prompt

# Get a memory for the conversational retriever


def get_memory(llm):
    memory = ConversationBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )
    return memory

# Get retrieval question and answer


def get_qa(llm, prompt, vectordb, memory):
    retriever = vectordb.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,

        verbose=True
    )

    return qa