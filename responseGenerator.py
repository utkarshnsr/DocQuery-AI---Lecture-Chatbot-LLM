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
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI


class ResponseGeneratorClass:

    def __init__(self):
        self.file = None
        self.fileText = "None"
        self.splittedData = None
        self.vectorDB = None

    
    def addFile(self,pdfFile):
        self.file = pdfFile

    def configureGemini(self):
        load_dotenv()
        API_KEY = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=API_KEY)

    def extractProcess(self):
        if self.file:
            pdfReader = PdfReader(self.file)
            self.fileText = ""
            for page in pdfReader.pages:
                self.fileText += page.extract_text()

    def chunkProcess(self):
        r_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splittedData = r_splitter.split_text(text=self.fileText)

    def embedProcess(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
        db = FAISS.from_texts(self.splittedData, embeddings)
        db.save_local("faiss_index")

    def initiateRAGProcess(self):
        self.configureGemini()
        self.extractProcess()
        self.chunkProcess()
        self.embedProcess()


    def conversational_chain(self):
        prompt_template = """
        You are a really helpful wonderful assistant!
        
        I want you to answer the question as detailed as possible given the context (make sure to provide all the details). 
        Most importantly, if the answer is not in given context just say, "answer is not available from the lecture notes you provided".
        MAKE SURE YOU DO NOT PROVIDE THE WRONG INFORMATION! 
        
        ALSO IF THEY SAY SOMETHING LIKE THANK YOU AND ARE BEING GRATEFUL FOR YOUR HELP THEN YOU DON'T NEED TO PROVIDE AN ANSWER FOR THAT. JUST SAY YOU ARE WELCOME OR SOMETHING LIKE THAT.
        
        \n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        geminiModel = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.1)
        #print(f"THE TYPE OF MODEL: {type(geminiModel)}")
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        chain = load_qa_chain(llm=geminiModel, chain_type="stuff", prompt=prompt)
        return chain

    def userInputProcess(self, question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
        vectorDB = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
        docs = vectorDB.similarity_search(question)
        chain = self.conversational_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True, )
        return response['output_text']