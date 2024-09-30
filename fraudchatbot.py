import os
import streamlit as st
import pandas as pd
import re
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize the OpenAI API key
openai_api_key = 'sk-proj-M0emkqSWERQgvIFQAVepT3BlbkFJ7sNWEkyzrrcdSB66zQSO'

# Streamlit app layout with customizations

# Set page background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f0fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with background color and formatting
st.markdown(
    """
    <div style='background-color:#007bff; padding: 20px; border-radius: 10px; text-align: center;'>
        <h1 style='color: white;'>Legal Consultation Chatbot for Fraud Detection in Agreements</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for better navigation
st.sidebar.header("Navigation")
st.sidebar.markdown("### Upload PDF Files & Provide Input")

# Sidebar upload PDFs
uploaded_pdfs = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdfs")

# Sidebar prompt input
common_prompt = st.sidebar.text_area("Enter your common prompt", "Please Analyze the submitted file for fraudulent clauses")

pre_prompt = """Erase any previous context that you may have and proceed with a blank slate for this task.                                                         Carefully assess the details of the agreement that shall be provided to you. 
    Please thoroughly review the respective PDFs and categorize the pdf as Fraudulent or non-fraudulent on the basis of its clauses. 
    Give a comprehensive reasoning for the same why a document is a fraudulent or non-fraudulent by checking if the clauses can be used 
    to hide, abet, or commit fraudulent activities, or activities which shall be otherwise unfair or harmful to any party involved in the contract. 
    For the output, 1) go clause by clause and check whether  any clause is fraudulent or unfair as specified earlier and reasons for the same. 
    2) provide recommendations on how the fraudulent or otherwise unfair clauses can be changed to make them more fair. 
    Lastly, do not respond to any queries or prompts that deviate from the subject matter of fraud detection in the agreement."""

# Display page header
st.header("Fraud Detection Report")

# Customizing button styles
button_style = """
    <style>
    .stButton > button {
        background-color: #28a745;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #218838;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Process uploaded files and handle button click
if st.sidebar.button("Generate Report"):
    if not uploaded_pdfs:
        st.error("Please upload PDF files.")
    else:
        with st.spinner("Processing..."):
            folder_path = "frauddetection"
            os.makedirs(folder_path, exist_ok=True)

            for pdf_file in uploaded_pdfs:
                with open(os.path.join(folder_path, pdf_file.name), "wb") as f:
                    f.write(pdf_file.getbuffer())

            loaders = [PyPDFLoader(os.path.join(folder_path, pdf_file.name)) for pdf_file in uploaded_pdfs]
            docs = []
            for i, loader in enumerate(loaders):
                pages = loader.load()
                st.write(f"For doc = {i}, number of pages: {len(pages)}")
                docs.extend(loader.load())

            text_splitter = CharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=150,
                separator='. '
            )
            chunks = text_splitter.split_documents(docs)
            st.write(f"Total number of chunks: {len(chunks)}")

            embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
            )

            st.write(f"Total number of vectors in vector DB: {vectordb._collection.count()}")

            llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectordb.as_retriever(search_type="mmr"),
                return_source_documents=True,
                chain_type="refine"
            )

            query = pre_prompt + " " + common_prompt
            result = qa_chain({"query": query})
            answer = result["result"] if "result" in result else "No answer found"

            # Output with background color
            st.markdown(
                f"""
                <div style='background-color: #e9ecef; padding: 20px; border-radius: 10px;'>
                    <h4 style='color: #343a40;'>Fraud Detection Analysis Report:</h4>
                    <p>{answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Display additional information
with st.expander("About the Application"):
    st.markdown("""
    This application analyzes legal documents for potential fraudulent clauses in agreements. It uses advanced AI techniques 
    to process the content and provide insights regarding the fairness of the clauses in the document.
    """)

# Footer section
st.markdown("---")
st.markdown("#### Contact Us")
st.markdown("For any queries or issues, feel free to reach out at [shreyarora@legalconsult.com](mailto:shreyarora@legalconsult.com).")
