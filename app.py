import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extractText()
    return text

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
                st.write(raw_txt)


if __name__=='__main__':
    main()