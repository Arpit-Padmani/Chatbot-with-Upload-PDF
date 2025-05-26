from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

# Load embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample text chunks (from PDF)
text_chunks = [
    "LangChain is a framework for developing applications powered by language models.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
]

# Build vector store
vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Define LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # or any compatible HF model
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

# Create the ConversationalRetrievalChain object
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True  # Optional: include document sources in output
)

print(conversation_chain)