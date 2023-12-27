# app.py
import utils
import streamlit as st

st.set_page_config(page_title="Chatbot-TFG-UBU", page_icon="💬")
st.header('💬 Resuelve tus dudas sobre el TFG')
st.write('Resulve tus dudas para la realización del TFG del Grado de Ingeniería Informática de la Universidad de Burgos.')
st.sidebar.image(
    "https://www.ubu.es/sites/all/themes/ubu_theme/images/UBUEscudo-1910.png?tok=Tq7E9hnJ", use_column_width=True)


class CustomDataChatbot:

    def setup_qa_chain(self):
        # Import vectordb
        vectordb = utils.import_vectordb()

        # Get our customized prompt
        prompt = utils.get_prompt()

        # Generate a question and asnwer based on our RAG
        qa_chain = utils.get_qa_chain(prompt, vectordb)

        return qa_chain

    @utils.enable_chat_history
    def main(self):

        # User Inputs
        user_input = st.chat_input(placeholder="Escribe aqui tus dudas")

        if user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input})

            # render the user's new message
            with st.chat_message("user"):
                st.markdown(user_input)

            qa_chain = self.setup_qa_chain()

            with st.chat_message("assistant"):
                with st.spinner("El Chatbot está escribiendo..."):

                    response = qa_chain.run(user_input)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})


# streamlit run app.py
if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
