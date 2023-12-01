
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
