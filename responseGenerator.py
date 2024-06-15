import pprint
import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings




def extractProcess(file):
    if file:
        pdfReader = PdfReader(file)
        text = ""
        for page in pdfReader.pages:
            text += page.extract_text()
        
        return text


def chunkProcess(text):
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    dataSplitted = r_splitter.split_text(text=text)
    return dataSplitted

def embedProcess(splittedData):
    configureGemini()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
    db = FAISS.from_texts(splittedData, embeddings)
    db.save_local("faiss_index")
    return db

def retrieveProcess(db, question):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)
    return docs






def configureGemini():
    load_dotenv()
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=API_KEY)
    return API_KEY


def main(inputPrompt):
    configureGemini()
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"For the question that is asked by the user please provide a gentle response addressing them by sir. Here is the question: {inputPrompt}"
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    configureGemini()
    main()