# app.py
import utils
import os
import streamlit as st
from streaming import StreamHandler

from typing import List, Union

from dotenv import load_dotenv, find_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

st.set_page_config(page_title="Chatbot-TFG-UBU", page_icon="💬")
st.header('💬 Resuelve tus dudas sobre el TFG')
st.write('Resulve tus dudas para la realización del TFG del Grado de Ingeniería Informática de la Universidad de Burgos.')
st.sidebar.image(
    "https://www.ubu.es/sites/all/themes/ubu_theme/images/UBUEscudo-1910.png?tok=Tq7E9hnJ", use_column_width=True)


class CustomDataChatbot:

    def __init__(self):
        self.llm = utils.get_llm
        st.session_state["llm"] = utils.get_llm

    def setup_qa_chain(self):
        # Import vectordb
        vectordb = utils.import_vectordb()

        # Get our customized prompt
        prompt = utils.get_prompt()

        # Get omemory for conversational retrieval
        memory = utils.get_memory(self.llm)

        # Generate a question and asnwer based on our RAG
        qa_chain = utils.get_qa_chain(self.llm, prompt, vectordb, memory)

        return qa_chain

    @utils.enable_chat_history
    def main(self):

        # User Inputs
        user_query = st.chat_input(placeholder="¿Como puedo ayudarte?")

        if user_query:
            qa_chain = self.setup_qa_chain()

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})


# streamlit run app.py
if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
