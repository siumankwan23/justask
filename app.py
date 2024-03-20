# app.py
import streamlit as st
from dotenv import load_dotenv
import pickle
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
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

    st.markdown('## Environment Variables')
    openai_api_key = st.text_input("Enter OPENAI_API_KEY")
    if openai_api_key:  # Check if the input is not empty
        os.environ["OPENAI_API_KEY"] = openai_api_key  # Set the environment variable

# Main function
def main():
    
    st.title("Embedding Loader and Question Answering")
    
    # Load embeddings
    embeddings = get_embeddings()
    if embeddings is None:
        st.stop()

    # Allow users to ask questions
    question = st.text_input("Ask a question:")
    if st.button("Submit"):
        if question:
            # Process user question using the embeddings and provide a response
            # response = process_question(question, embeddings)
            # get the docs related to the question
            docs = retrieve_docs(question, embeddings)
            response = generate_response(docs, question)
            st.write("Response:", response)
        else:
            st.warning("Please enter a question.")

def retrieve_docs(question, embeddings):
    docs = embeddings.similarity_search(question, k=3)
    if len(docs) == 0:
        raise Exception("No documents found")
    else:
        return docs

def generate_response(docs, question):
    llm = ChatOpenAI(temperature=0.0, max_tokens=1000, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(cb)
    return response

# Function to process user question and provide a response
def process_question(question, embeddings):
    # Perform necessary operations with embeddings to generate a response
    # Example: Find similar embeddosings, use a pre-trained model, etc.
    # For demonstration purposes, let's just return a dummy response
    return "This is a dummy response to the question: '{}'".format(question)

# Function to get embeddings
def get_embeddings():
    root_dir = os.path.dirname(__file__)
    embeddings_paths = "/mount/src/justask/s.pkl"
    embeddings_path = os.path.join(root_dir, "s.pkl")
    #embeddings_paths = os.path.join(root_dir, "c.jpg")
    #st.image('c.jpg', caption='Sunrise by the mountains')
    st.write(embeddings_paths)
    st.write(embeddings_path)

    #embeddings_path = "s_embeddings.pkl"  # Path to your embeddings file
    if os.path.exists(embeddings_path):
        embeddings = load_embeddings(embeddings_path)
        st.write("Embeddings loaded successfully!")
        return embeddings
    else:
        st.error("Embeddings file not found!")
        st.write(embeddings_path)
        return None

# Function to load embeddings from a pickle file
def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings
    
if __name__ == "__main__":
    main()
