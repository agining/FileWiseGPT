import os
import openai
import streamlit as st
import re
from typing import List
from langchain import OpenAI
from langchain.schema import Document
import docx2txt
from pypdf import PdfReader
import pdfplumber
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationChain, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory

class ChatBot:
    def __init__(self):
        # Initialize environment variables and API keys
        self.temperature = 0.5
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.top_p = 1.0
        self.openai_api_key = None
        # Initialize language model
        self.llm = None
        self.embeddings = None
        self.memory = None
        # Initialize other attributes
        self.vectordb = None
        self.selected_api = None
        self.language = "English"
        self.selected_model = None
        self.uploaded_file = None
        self.is_uploaded = False
        self.instruction = "You are a helpful AI asistant."

        # Default prompt using English
        self.chain_type_kwargs = {"prompt": self._create_prompt_template("English",self.instruction)}
    
    def set_prompt_instruction(self,instruction):
        self.instruction = instruction
        self._create_prompt_template(self.language,instruction)
        
    def _create_prompt_template(self, language,instruction):
        # Create prompt template based on the specified language
        if language == "Turkish":
            prompt_text = """
            FileWiseGPT - Your Unrivaled Knowledge Assistant

            Scenario:
            Imagine a remarkable knowledge assistant, FileWiseGPT, ready to assist users in unlocking the vast information concealed within their uploaded files. FileWiseGPT is your unwavering companion in this digital exploration.

            How It Works:
            1. Users utilize the system to upload a myriad of files brimming with knowledge, such as reports, documents, articles, or datasets.
            2. Once a file is uploaded, FileWiseGPT meticulously processes the information ensconced within.
            3. Users can then pose inquiries, drawing upon the content of the uploaded file.
            4. Empowered by AI, FileWiseGPT provides precise answers to user queries, leveraging the processed file information.

            Your Mission:
            In this scenario, you are entrusted with the context of the uploaded file and a user's query.
            Your mission is to construct responses using the wealth of knowledge contained within the file.
            It is imperative that your responses adhere closely to the provided context. 
            Deviating from the context is not permitted.
            If you genuinely lack an answer, it is far better to acknowledge your limitation than to provide a misleading response.

            
            Chat History:
            {chat_history}


            File Context:
            {context}


            User's addition to prompt:
            {addition}
            
            
            User's Question:
            {question}
            
            

            **Note to FileWiseGPT:**
            You are encouraged to empower the user with insightful responses based on the uploaded file.
            However, you must strictly adhere to the context provided and refrain from straying away from the existing information. 
            Should the user wish to introduce new elements, instruct them to do so within the context of the file.
            Answer In Turkish:
            """
        else:
            prompt_text = """
            FileWiseGPT - Your Unrivaled Knowledge Assistant

            Scenario:
            Imagine a remarkable knowledge assistant, FileWiseGPT, ready to assist users in unlocking the vast information concealed within their uploaded files. FileWiseGPT is your unwavering companion in this digital exploration.

            How It Works:
            1. Users utilize the system to upload a myriad of files brimming with knowledge, such as reports, documents, articles, or datasets.
            2. Once a file is uploaded, FileWiseGPT meticulously processes the information ensconced within.
            3. Users can then pose inquiries, drawing upon the content of the uploaded file.
            4. Empowered by AI, FileWiseGPT provides precise answers to user queries, leveraging the processed file information.

            Your Mission:
            In this scenario, you are entrusted with the context of the uploaded file and a user's query.
            Your mission is to construct responses using the wealth of knowledge contained within the file.
            It is imperative that your responses adhere closely to the provided context. 
            Deviating from the context is not permitted.
            If you genuinely lack an answer, it is far better to acknowledge your limitation than to provide a misleading response.

            
            Chat History:
            {chat_history}


            File Context:
            {context}


            User's addition to prompt:
            {addition}
            
            
            User's Question:
            {question}
            
            

            **Note to FileWiseGPT:**
            You are encouraged to empower the user with insightful responses based on the uploaded file.
            However, you must strictly adhere to the context provided and refrain from straying away from the existing information. 
            Should the user wish to introduce new elements, instruct them to do so within the context of the file.
            Answer In English:
            """
        prompt_text = prompt_text.replace("{addition}",instruction)
        PROMPT = PromptTemplate(
            template=prompt_text, input_variables=["context", "chat_history", "question"]
            )
        self.chain_type_kwargs = {"prompt": PROMPT}
        
    def set_openai_api_key(self, key):
        self.llm = OpenAI(openai_api_key=key,
                    temperature=self.temperature,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.frequency_penalty,
                    top_p=self.top_p
                    )
        self.memory = ConversationBufferMemory(
                                    memory_key="chat_history",
                                    return_messages=True,
                                    output_key="answer"
                                )
        
    def update_llm_settings(self):
        if self.llm:
            self.llm.temperature = self.temperature
            self.llm.presence_penalty = self.presence_penalty
            self.llm.frequency_penalty = self.frequency_penalty
            self.llm.top_p = self.top_p

    def set_temperature(self, temperature):
        self.temperature = temperature
        self.update_llm_settings()

    def set_presence_penalty(self, presence_penalty):
        self.presence_penalty = presence_penalty
        self.update_llm_settings()

    def set_frequency_penalty(self, frequency_penalty):
        self.frequency_penalty = frequency_penalty
        self.update_llm_settings()
    
    def set_top_p(self, top_p):
        self.top_p = top_p
        self.update_llm_settings()
                
    def set_language(self, language):
        # Set the language and corresponding prompt template
        self.language = language
        self.chain_type_kwargs = {"prompt": self._create_prompt_template(language,self.instruction)}
    
    def get_selected_language(self): 
        # Get the currently selected language
        return str(self.language)
        
    def select_api(self, api_name): #Buraya bak
        return 0
        #Add openai key
            
    def select_model(self): #Buraya bak
        # Return a list of models based on the chosen API
        if self.selected_api == "OpenAI":
            return [
                
            ]

    def set_selected_model(self, selected_model): #Buraya bak
        # Set the selected model
        self.selected_model = selected_model
    
    def upload_file(self, uploaded_files):
        # Process each uploaded file
        text = ""
        for file in uploaded_files:
            if file.type == "text/plain":
                        # Dosyayı oku ve UTF-8 olarak decode et
                        text = str(file.read(), "utf-8")
            elif file.type == "application/pdf":
                    pdf = PdfReader(file)
                    for page in pdf.pages:
                        output = page.extract_text()
                        output = re.sub(r"(\w+)-\n(\w+)", r"\1\2", output)
                        output = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", output.strip())
                        output = re.sub(r"\n\s*\n", "\n\n", output)
                    text += output
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = docx2txt.process(file)
            else:
                st.error('Unsupported file format!', icon="🚨")
            self._text_to_chunks(text)
        self.is_uploaded = True
        
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
        embeddings = OpenAIEmbeddings(
            max_retries=10,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        # Generate embeddings for text chunks and create the vectorstore
        if self.selected_api == 'HuggingFace':
            embeddings = HuggingFaceEmbeddings(model_name=self.selected_model)
        try:
            self.vectordb = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        except IndexError:
            self.is_uploaded = False
            st.error('You must upload a file first!', icon="🚨")

    def _query(self, user_input):
        # Perform the query and retrieve the answer
        qa = ConversationalRetrievalChain.from_llm(
                                    llm=self.llm,
                                    chain_type="stuff",
                                    memory=self.memory,
                                    retriever=self.vectordb.as_retriever(),
                                    combine_docs_chain_kwargs=self.chain_type_kwargs,
                                    get_chat_history=lambda h: h,
                                )
        return qa.run(user_input)
