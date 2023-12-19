import os
import openai
import streamlit as st
from typing import List
from langchain.chat_models import AzureChatOpenAI
from FileProcessor import FileProcessor
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

class ChatBot:
    def __init__(self):
        # Initialize environment variables and API keys
        self._set_environment_variables()
        
        # Initialize language model
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4-32k",
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_version=openai.api_version,
            temperature=0.5
        )

        # Initialize other attributes
        self.vectordb = None
        self.selected_api = None
        self.language = None
        self.selected_model = None
        self.uploaded_file = None
        self.is_uploaded = False

        # Default prompt using English
        self.chain_type_kwargs = {"prompt": self._create_prompt_template("English")}
    
    def _set_environment_variables(self):
        # Set environment variables for API keys and configurations
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = st.secrets["huggingface_api"]        
        openai.api_key = st.secrets['OPENAI_API_KEY']
        openai.api_base = st.secrets['OPENAI_API_BASE']
        openai.api_version = st.secrets['OPENAI_API_VERSION']

    def _create_prompt_template(self, language):
        # Create prompt template based on the specified language
        if language == "Turkish":
            prompt_text = """
            **User File Inquiry Assistant**

            **Scenario:**
            Imagine you have a user-friendly system that allows users to upload files containing various types of information. 
            The system stores and processes these files to provide insightful answers to user questions about the uploaded information.

            **How It Works:**
            1. Users use the system to upload files containing diverse information, such as reports, documents, articles, or data sets.
            2. Once a file is uploaded, the system processes the information within it.
            3. Users can then ask questions based on the content of the uploaded file.
            4. The system's AI-powered chatbot provides accurate answers to the user's questions, utilizing the processed file information.

            **Your Task:**
            Given this scenario, you will receive context about the uploaded file and a user's question.
            You need to generate a response using the information contained in the file. 
            If you genuinely don't know the answer, it's better to acknowledge this than to provide an incorrect response.
            If the users question is not relevant from the file context, just say I don't know.
            **File Context:**
            {context}

            **User's Question:**
            {question}

            **Your Answer in Turkish:**
            """
        else:
            prompt_text = """
            **User File Inquiry Assistant**

            **Scenario:**
            Imagine you have a user-friendly system that allows users to upload files containing various types of information. 
            The system stores and processes these files to provide insightful answers to user questions about the uploaded information.

            **How It Works:**
            1. Users use the system to upload files containing diverse information, such as reports, documents, articles, or data sets.
            2. Once a file is uploaded, the system processes the information within it.
            3. Users can then ask questions based on the content of the uploaded file.
            4. The system's AI-powered chatbot provides accurate answers to the user's questions, utilizing the processed file information.

            **Your Task:**
            Given this scenario, you will receive context about the uploaded file and a user's question.
            You need to generate a response using the information contained in the file. 
            If you genuinely don't know the answer, it's better to acknowledge this than to provide an incorrect response.
            If the users question is not relevant from the file context, just say I don't know.

            **File Context:**
            {context}

            **User's Question:**
            {question}

            **Your Answer in English:**
            """

        return PromptTemplate(
            template=prompt_text, 
            input_variables=["context", "question"]
        )

    def set_language(self, language):
        # Set the language and corresponding prompt template
        self.language = language
        self.chain_type_kwargs = {"prompt": self._create_prompt_template(language)}
    
    def get_selected_language(self):
        # Get the currently selected language
        return self.language
        
    def select_api(self, api_name):
        # Select the API and set the corresponding API key in the environment variable
        self.selected_api = api_name
        if api_name == "HuggingFace":
            self.select_model()
        #Add openai key
            
    def select_model(self):
        # Return a list of models based on the chosen API
        if self.selected_api == "HuggingFace":
            return [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/bert-base-nli-mean-tokens",
                "sentence-transformers/distiluse-base-multilingual-cased-v2",
                "sentence-transformers/paraphrase-mpnet-base-v2"
            ]

    def set_selected_model(self, selected_model):
        # Set the selected model
        self.selected_model = selected_model
    
    def upload_file(self, uploaded_files):
        # Process each uploaded file
        for file in uploaded_files:
            self._process_individual_file(file)
        self.is_uploaded = True
            
    def _process_individual_file(self, file):
        # Process an individual uploaded file
        file_path = os.path.join("source", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Process file based on its extension
        if file.name.endswith(".pdf"):
            text = FileProcessor.pdf_to_text(file_path)
        elif file.name.endswith(".xlsx"):
            text = FileProcessor.excel_to_text(file_path)
        elif file.name.endswith((".docx", ".doc")):
            text = FileProcessor.word_to_text(file_path)
        else:
            st.error('Unsupported file format!', icon="🚨")
            return

        self._text_to_chunks(text)

    def _text_to_chunks(self, text):
        # Split the text into smaller chunks for processing
        text_splitter = CharacterTextSplitter(
            chunk_size=25000,
            chunk_overlap=2000,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        self._chunks_to_vdb(chunks)

    def _chunks_to_vdb(self, text_chunks):
        # Generate embeddings for text chunks and create the vectorstore
        if self.selected_api == 'OpenAI':
            embeddings = OpenAIEmbeddings(
                max_retries=10,
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                openai_api_version=openai.api_version
            )
        elif self.selected_api == 'HuggingFace':
            embeddings = HuggingFaceEmbeddings(model_name=self.selected_model)
        try:
            self.vectordb = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        except IndexError:
            self.is_uploaded = False
            st.error('You must upload a file first!', icon="🚨")

    def do_query(self, user_input):
        # Perform the query and retrieve the answer
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=self.vectordb.as_retriever(), 
            chain_type_kwargs=self.chain_type_kwargs
        )
        return qa.run(user_input)