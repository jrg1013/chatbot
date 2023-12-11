# chatbot

Desarrollo de un chatbot basado en 'Large Language Model' (LLM) y técnicas de 'Retrieval-augmented Generation' (RAG) para asistente de dudas sobre la realización del TFG en el Grado de Ingeniería Informática

## Descripción

"El proceso de realización del TFG puede generar múltiples dudas en los alumnos, desde la elección del tema, la asignación de tutores, los diferentes pasos a seguir, documentación final a entregar o la presentación final. Para asistir en este proceso, se propone el desarrollo de un chatbot basado en modelos de lenguaje grandes (LLM) como ChatGPT, Bard o LLama. Para mejorar la precisión de la respuesta del modelo y evitar imprecisiones o alucinaciones se utilizará técnicas de “Retrieval-Augmented generation” (RAG), que permiten enriquecer las instrucciones al modelo con información específica sobre el TFG debidamente embebida en vectores (en el espacio del modelo) y almacenada en una base de datos vectorizada. Además, se quiere comparar el desempeño de este chatbot con un chatbot preexistente desarrollado en anteriores TFG, y determinar las mejoras y ventajas de esta nueva aproximación.
Objetivos: Diseñar y desarrollar un chatbot basado en “Large Language Model” (LLM) que pueda responder a las dudas de los/las estudiantes sobre la realización del TFG. Integrar técnicas de ""Retrieval-Augmented Generation"" (RAG) para mejorar la precisión y relevancia de las respuestas del Chatbot, aprovechando la información contenida en documentos existentes sobre el TFG. Comparar el rendimiento y eficacia del chatbot desarrollado con un chatbot preexistente creado en un TFG anterior. "

## Trabajos teóricos relaccionados

https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

## Trabajos prácticos relaccionados

https://github.com/aav0038/UBU-CHATBOT
https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/
https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/

## Instalación entorno

sh setup_env.sh

## Setup envioroment

# Activate venev

source venv/bin/activate

# Install

pip install -r path/to/requirements.txt

# Export

pip freeze > requirements.txt

## Run de la APP

# Vectorización de información

python learn.py

# Chatbot

streamlit run app.py
