from langchain.vectorstores import Epsilla
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st

import subprocess
from typing import List

# Customize the layout
st.set_page_config(page_title="Chatbot-UBU-TFG", layout="wide", )

# The 1st welcome message
st.title("💬 Resuelve tus dudas sobre el TFG")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "asistente", "content": "¿Como puedo ayudarte?"}]

# A fixture of chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Local embedding model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db_path = "./faiss_index_hp"

vectordb = FAISS.load_local(
    db_path,
    embeddings
)

# Answer user question upon receiving
if question := st.chat_input():
    st.session_state.messages.append({"role": "usuario", "content": question})

    context = '\n'.join(map(lambda doc: doc.page_content,
                        vectordb.similarity_search(question)))

    st.chat_message("usuario").write(question)

    # Here we use prompt engineering to ingest the most relevant pieces of chunks from knowledge into the prompt.
    prompt = f'''
    Answer the Question based on the given Context. Try to understand the Context and rephrase them.
    Please don't make things up or say things not mentioned in the Context. Ask for more information when needed. 
    Responde en el mismo idioma en el que se te pregunta. Eres un Chatbot para la resolución de dudas del Trabajo fin de Grado
    del Grado de Ingeniería Informática de la Universidad de Burgos

    Context:
    {context}

    Question:
    {question}

    Answer:
    '''
    print(prompt)

    # Call the local LLM and wait for the generation to finish. This is just a quick demo and we can improve it
    # with better ways in the future.
    command = ['llm', '-m', 'mistral-7b-openorca', prompt]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    content = ''
    while True:
        output = process.stdout.readline()
        if output:
            content = content + output
        return_code = process.poll()
        if return_code is not None:
            break

    # Append the response
    msg = {'role': 'asistente', 'content': content}
    st.session_state.messages.append(msg)
    st.chat_message("asistente").write(msg['content'])
