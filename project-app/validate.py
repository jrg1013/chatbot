# validate.py

import utils
import langchain

# Documents loaders
from langchain import document_loaders

# Prompts
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Evaluation
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
from langchain.llms import HuggingFaceHub
from langchain.evaluation import load_evaluator


loaders = [
    document_loaders.CSVLoader(
        file_path="./documents/Preguntas-Respuestas - ONLINE.csv",
        csv_args={
            "delimiter": ";",
            "quotechar": '"',
            "fieldnames": ["Intent", "Ejemplo mensaje usuario", "Respuesta"],
        })

]

documents = []

for loader in loaders:
    documents.extend(loader.load())

# Get llm
llm = utils.get_llm()

# Import vectordb
vectordb = utils.import_vectordb()

# Get our customized prompt
prompt = utils.get_prompt()

# Generate a question and asnwer based on our RAG
qa_chain = utils.get_qa_chain(prompt, vectordb)

# Hard-coded examples
examples = [
    {
        "query": "¿Se pueden adjuntar videos en el depósito del TFG?",
        "answer": "No se pueden adjuntar videos en el depósito. Se subirá un \
        documento con los enlaces a dichos vídeos que deberán estar colgados en YouTube."
    },
    {
        "query": "¿Cómo se hace la defensa del TFG?",
        "answer": "La defensa se realiza oralmente, apoyándose en medios audiovisuales y \
        demostraciones prácticas, durante un periodo de tiempo de 10 a 15 minutos. \
        Posteriormente se realiza un turno de preguntas por parte del tribunal a responder \
        por el alumnado."
    },

]

# Manual evaluation
langchain.debug = True
qa_chain.run(examples[0]["query"])
# Turn off the debug mode
langchain.debug = False

# Automatic QA
predictions = qa_chain.apply(examples)
# eval_chain = QAEvalChain.from_llm(llm)
