# app.py
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
# get callback to get stats on query cost
from langchain.callbacks import get_openai_callback
import os
load_dotenv() 
# Sidebar contents
with st.sidebar:
    st.title('💬 PDF Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered PDF chatbot built using:
    - [Streamlit](https://streamlit.io/) Frontend Framework
    - [LangChain](https://python.langchain.com/) App Framework
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [FAISS](https://github.com/facebookresearch/faiss) vector store
 
    ''')
    add_vertical_space(5)
    st.write('Made by [William Gillett](https://github.com/wmgillett/chat-pdf-langchain-faiss-streamlit)')

def main():
    st.write("Hello")
    st.header("Chat with PDF 💬")
    # upload a PDF file
    pdf = upload_pdf()
    # check for pdf file
    if pdf is not None:
        # process text in pdf and convert to chunks
        chunks = process_text(pdf)
        # get the embeddings
        vector_store = get_embeddings(chunks, pdf)
        # ask the user for a question
        question = st.text_input("Ask a question")
        if question:
            # get the docs related to the question
            docs = retrieve_docs(question, vector_store)
            response = generate_response(docs, question)
            st.write(response)

# upload a pdf file from website
def upload_pdf():
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    return pdf

# convert the pdf to text chunks
def process_text(pdf):
    pdf_reader = PdfReader(pdf)
    # extract the text from the PDF
    page_text = ""
    for page in pdf_reader.pages:
        page_text += page.extract_text()
    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=page_text)
    return chunks

# find or create the embeddings
def get_embeddings(chunks, pdf):
    store_name = pdf.name[:-4]
    # check if vector store already exists
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
        st.write("Embeddings loaded from disk")
    #else create embeddings and save to disk
    else:
        embeddings = OpenAIEmbeddings()
        # create vector store to hold the embeddings
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        # save the vector store to disk
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        st.write("Embeddings saved to disk")
    return vector_store

# retrieve the docs related to the question
def retrieve_docs(question, vector_store):
    docs = vector_store.similarity_search(question, k=3)
    # st.write(docs)
    return docs 

# generate the response
def generate_response(docs, question):
    llm = OpenAI(temperature=0.0, max_tokens=1000, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(cb)
    return response






if __name__ == '__main__':
    main()