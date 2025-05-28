import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import  css,bot_template,user_template
from langchain_google_genai import ChatGoogleGenerativeAI
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
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embedding)
    return vectorstore

def get_conversation_chain(vector):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
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
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history=response['chat_history']
        for i,message in enumerate(st.session_state.chat_history):
            if i%2==0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process a PDF first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot With PDF",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if 'conversation' not in st.session_state :
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None
    st.header("ChatBot With PDF :books:")
    user_quetion=st.text_input('Ask Question about PDF')
    if user_quetion:
        handle_userinput(user_quetion)

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