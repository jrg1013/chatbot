# app.py
from typing import List, Union

import cfg
import tokens
import utils

from dotenv import load_dotenv, find_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
import glob
import textwrap
import time

import langchain

# loaders
from langchain import document_loaders

# splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

# prompts
from langchain.prompts import PromptTemplate

# chain
from langchain.chains import LLMChain

# vector stores
from langchain.vectorstores import FAISS

# models
from langchain.llms import HuggingFacePipeline
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub

# retrievers
from langchain.chains import RetrievalQA

# evaluation
from langchain.evaluation.qa import QAGenerateChain

import torch

import streamlit as st

import subprocess
from typing import List

# Acceso al llm
huggingfacehub_api_token = tokens.huggingfacehub_api_token

llm = utils.get_llm()

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

print(vectordb.similarity_search('videos'))

question = "Give me 5 examples of magic potions and explain what they do? "

template = """<s>[INST] You are given the context after <<CONTEXT>> and a question after <<QUESTION>>.

Answer the question by ONLY using the information in <<CONTEXT>>. Only base your answer on the information in the <<CONTEXT>>.

Answer in the same language as the <<QUESTION>>. Responde en el mismo idioma que la pregunta.

<<QUESTION>>{question}\n
<<CONTEXT>>{context} 

[/INST]

"""

prompt = PromptTemplate(template=template, input_variables=[
                        "question", "context"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question=question, context=""))
