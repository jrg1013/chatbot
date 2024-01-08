# util.py

import cfg
import tokens
import streamlit as st

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


def import_vectordb():
    # Import vector database
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

    return vectordb


def get_llm():
    huggingfacehub_api_token = tokens.huggingfacehub_api_token

    llm = HuggingFaceHub(
        repo_id=cfg.model_name,
        model_kwargs={
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "max_length": cfg.max_length,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "do_sample": cfg.do_sample,
            "num_return_sequences": cfg.num_return_sequences
        },
        huggingfacehub_api_token=huggingfacehub_api_token
    )

    return llm


def get_prompt():
    # Get Prompt based in the context and template in config
    template = cfg.template
    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])
    return prompt


def get_qa_chain(prompt, vectordb):

    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"k": 1, "score_threshold": .0})

    # Get the llm
    llm = get_llm()

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )

    return qa_chain


def enable_chat_history(func):

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "¿Como puedo ayudarte?"}]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)
