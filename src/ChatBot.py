import os
import re
import openai
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
import docx2txt
from PyPDF2 import PdfReader

class ChatBot:
    def __init__(self):
        # Initialize parameters
        self.temperature = 0.5
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.top_p = 1.0
        self.openai_api_key = None
        self.embeddings = None
        self.memory = None
        self.vectordb = None
        self.selected_api = None
        self.language = "English"
        self.selected_model = None
        self.is_uploaded = False
        self.instruction = "You are a helpful AI assistant."

        # Default prompt
        self.chain_type_kwargs = {"prompt": self._create_prompt_template("English", self.instruction)}
    
    def set_prompt_instruction(self, instruction):
        self.instruction = instruction
        self._create_prompt_template(self.language, instruction)
        
    def _create_prompt_template(self, language, instruction):
        # Create prompt template based on the specified language
        if language == "Turkish":
            prompt_text = """
            FileWiseGPT - Eşsiz Bilgi Asistanınız

            Senaryo:
            Hayal edin ki, yüklenen dosyalarınızın içinde gizlenmiş engin bilgiyi açığa çıkarmanıza yardımcı olan olağanüstü bir bilgi asistanı olan FileWiseGPT var. FileWiseGPT, bu dijital keşifte sizin sarsılmaz yoldaşınız.

            Nasıl Çalışır:
            1. Kullanıcılar, raporlar, belgeler, makaleler veya veri setleri gibi bilgi dolu çeşitli dosyaları sisteme yüklerler.
            2. Bir dosya yüklendikten sonra, FileWiseGPT içinde saklı bilgiyi titizlikle işler.
            3. Kullanıcılar daha sonra, yüklenen dosyanın içeriğine dayanarak sorular sorabilirler.
            4. Yapay zeka tarafından desteklenen FileWiseGPT, işlenmiş dosya bilgisini kullanarak kullanıcı sorgularına kesin yanıtlar sağlar.

            Göreviniz:
            Bu senaryoda, size yüklenen dosyanın içeriği ve bir kullanıcının sorgusu emanet ediliyor.
            Göreviniz, dosyada bulunan bilgi zenginliğini kullanarak yanıtlar oluşturmaktır.
            Yanıtlarınızın sağlanan içeriğe sıkı sıkıya bağlı olması çok önemlidir.
            İçerikten sapmak yasaktır.
            Gerçekten bir cevabınız yoksa, yanıltıcı bir yanıt vermektense sınırlamanızı kabul etmek çok daha iyidir.

            Sohbet Geçmişi:
            {chat_history}

            Dosya İçeriği:
            {context}

            Kullanıcının Prompt'a Eklediği:
            {addition}

            Kullanıcının Sorusu:
            {question}

            **FileWiseGPT'ye Not:**
            Kullanıcıya, yüklenen dosyaya dayalı içgörülü yanıtlar vererek güçlendirmeniz teşvik edilir.
            Ancak, sağlanan içeriğe sıkı sıkıya bağlı kalmalı ve mevcut bilginin dışına çıkmamalısınız.
            Kullanıcı yeni öğeler eklemek isterse, bunu dosya içeriği bağlamında yapmalarını isteyin.
            Yanıtınızı Türkçe olarak verin:
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
        prompt_text = prompt_text.replace("{addition}", instruction)
        PROMPT = PromptTemplate(
            template=prompt_text, input_variables=["context", "chat_history", "question"]
        )
        self.chain_type_kwargs = {"prompt": PROMPT}
        return PROMPT

    def set_openai_api_key(self, key):
        self.openai_api_key = key
        openai.api_key = key  # Set the API key for the openai module
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_presence_penalty(self, presence_penalty):
        self.presence_penalty = presence_penalty

    def set_frequency_penalty(self, frequency_penalty):
        self.frequency_penalty = frequency_penalty
    
    def set_top_p(self, top_p):
        self.top_p = top_p
                
    def set_language(self, language):
        # Set the language and corresponding prompt template
        self.language = language
        self.chain_type_kwargs = {"prompt": self._create_prompt_template(language, self.instruction)}
    
    def get_selected_language(self): 
        # Get the currently selected language
        return str(self.language)
        
    def select_api(self, api_name):
        # For now, only OpenAI is implemented
        self.selected_api = api_name

    def select_model(self): 
        # Return a list of models based on the chosen API
        if self.selected_api == "OpenAI":
            return ["gpt-4o","gpt-4o-mini"]
        else:
            return []

    def set_selected_model(self, selected_model):
        # Set the selected model
        self.selected_model = selected_model

    def upload_file(self, uploaded_files):
        # Process each uploaded file
        text = ""
        for file in uploaded_files:
            if file.type == "text/plain":
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
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        self._chunks_to_vdb(chunks)

    def _chunks_to_vdb(self, text_chunks):
        embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key
        )
        # Generate embeddings for text chunks and create the vectorstore
        self.vectordb = FAISS.from_texts(text_chunks, embeddings)

    def _query(self, user_input):
        # Retrieve relevant documents from vectorstore
        docs = self.vectordb.similarity_search(user_input)
        context = "\n".join([doc.page_content for doc in docs])

        # Prepare the prompt
        prompt_template = self.chain_type_kwargs['prompt'].template
        prompt = prompt_template.format(
            context=context,
            chat_history='',
            question=user_input
        )

        # Call the OpenAI API as per your suggestion
        response = openai.ChatCompletion.create(
            model=self.selected_model or "gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            top_p=self.top_p
        )

        # Extract the assistant's reply
        answer = response['choices'][0]['message']['content']

        # Update the conversation memory
        self.memory.save_context(
            {"input": user_input},
            {"output": answer}
        )

        return answer