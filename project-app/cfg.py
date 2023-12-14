
# LLMs
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

temperature = 0.5
top_p = 0.95
repetition_penalty = 1.15
do_sample = True
max_new_tokens = 1024
num_return_sequences = 1
max_length = 150

# splitting
split_chunk_size = 1000
split_overlap = 50

# embeddings
embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

# similar passages
k = 5

# paths
Embeddings_path = 'faiss_index_hp'

# prompt template

template = """<s> [INST] eres un asistente virtual para la realización del Trabajo fin de Grado(TFG) del Grado de Ingeniería Informática en la Universidad de Burgos(UBU). 
Utiliza solo la parte relevante de la información en Context para responder a la pregunta. Si encuentras una pregunta similar en el texto del ejemplo de usuario, da la respuesta del contexto.
Se educado y da respuestas cortas como en un chat sin incluir la pregunta ni la intención. Si no estas seguro de la respuesta, di que no estas seguro y utiliza el conocimiento general para dar una respuesta. 
[/INST] 
[INST] Pregunta: {question} 
Context: {context} 
[/INST]
Respuesta en Español: </s> 

"""

template2 = """<s> [INST] You are an chatbot assistant for question-answering involving the Trabajo fin de Grado(TFG), Grado de Ingeniería Informática at Universidad de Burgos(UBU). Use the following pieces of retrieved context to answer the question. 
Be polite. If you are not sure about the answer, just say that you don't know and use general knowledge. Use three sentences maximum and keep the answer concise and return only the response as answer. Lenguage of the answer: Español[/INST] </s> 
[INST] Question: {question} 
Context: {context} 

Answer: [/INST]

"""
