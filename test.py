from InstructorEmbedding import INSTRUCTOR
import faiss
import numpy as np

# Load the model
model = INSTRUCTOR('hkunlp/instructor-xl')

# Define instructions and texts
instructions = ["Represent the document for retrieval"] * 2
texts = ["LangChain makes LLMs more usable", "ChatGPT is useful"]

# Prepare inputs for encoding
inputs = list(zip(instructions, texts))

# Get embeddings
embeddings = model.encode(inputs)

# Print the result
print("Shape of embeddings:", embeddings.shape)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Search
query_instruction = "Represent the query for retrieving relevant documents"
query_embedding = model.encode([[query_instruction, "How is ChatGPT used?"]])
D, I = index.search(np.array(query_embedding), k=1)

print("Most relevant index:", I[0][0])
