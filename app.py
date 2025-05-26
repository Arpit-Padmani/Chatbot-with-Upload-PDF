import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplate import  css,bot_template,user_template
from huggingface_hub import InferenceClient
from langchain.llms import GooglePalm
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
import os

from langchain_community.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_spiltter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_spiltter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    load_dotenv()
    # embedding = OpenAIEmbeddings(openai_api_type=os.getenv("OPENAI_API_KEY"))
    # embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embedding)
    return vectorstore

def get_conversation_chain(vector):
    # llm = GooglePalm()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    # llm = HuggingFaceHub(
    #     repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    #     model_kwargs={"temperature": 0.5, "max_length": 512}
    # )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",  # Updated model name
        temperature=0.5,
        google_api_key=os.getenv("GEMMINI_API_KEY")
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vector.as_retriever(),
                                                               memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    st.write(user_question)
    if st.session_state.conversation is not None:
        st.write(st.session_state.conversation)
        response = st.session_state.conversation({'question': user_question})
        st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process a PDF first.")


def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot With PDF",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if 'conversation' not in st.session_state :
        st.session_state.conversation = None

    st.header("ChatBot With PDF :books:")
    user_quetion=st.text_input('Ask Question about PDF')
    if user_quetion:
        handle_userinput(user_quetion)

    st.write(user_template.replace("{{MSG}}","Hello Robot"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello User"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader('Upload PDF Or Documents',accept_multiple_files=True)
        if st.button('Proccess'):
            with st.spinner("Proccessing"):

                raw_txt=get_pdf_text(pdf_docs)

                text_chunks=get_text_chunks(raw_txt)

                vector=get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vector)
                st.success("PDFs Processed")


if __name__=='__main__':
    main()