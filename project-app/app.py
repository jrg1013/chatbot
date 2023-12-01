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
llm = utils.get_llm()

# Import vectordb
embeddings, vectordb = utils.import_vectordb()

# Get our customized prompt
prompt = utils.get_prompt()

# Get omemory for conversational retrieval
memory = utils.get_memory(llm)

# Generate a question and asnwer based on our RAG
qa = utils.get_qa(llm, prompt, vectordb, memory)

question = "Mi nombre es Jose. Acuerdate de mi nombre."
result = qa({"question": question})
print(result["answer"])

question = "Como me llamo?"
result = qa({"question": question})
print(result["answer"])
