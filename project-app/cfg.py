
# LLMs
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

temperature = 0.5
top_p = 0.95
repetition_penalty = 1.15
do_sample = True
max_new_tokens = 1024
num_return_sequences = 1

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
template = """<s> [INST] You are an chatbot assistant for question-answering involving the Trabajo fin de Grado(TFG), Grado de Ingeniería Informática at Universidad de Burgos(UBU). Use the following pieces of retrieved context to answer the question. 
Be polite. If you are not sure about the answer, just say that you don't know and use general knowledge. Use three sentences maximum and keep the answer concise and return only the response as answer. Lenguage of the answer: Español[/INST] </s> 
[INST] Question: {question} 
Context: {context} 

Answer: [/INST]

"""
