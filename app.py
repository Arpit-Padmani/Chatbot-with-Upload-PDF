import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings,OpenAIEmbeddings
# from langchain.vectorstores import FAISS
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
        separator="/n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_spiltter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    load_dotenv()
    embedding = OpenAIEmbeddings(openai_api_type=os.getenv("OPENAI_API_KEY"))
    # embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embedding)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot With PDF",page_icon=":books:")

    st.header("ChatBot With PDF :books:")
    st.text_input('Ask Question about PDF')

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader('Upload PDF Or Documents',accept_multiple_files=True)
        if st.button('Proccess'):
            with st.spinner("Proccessing"):

                raw_txt=get_pdf_text(pdf_docs)

                text_chunks=get_text_chunks(raw_txt)

                vector=get_vectorstore(text_chunks)




if __name__=='__main__':
    main()