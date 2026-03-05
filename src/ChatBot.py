import os
import re
import json
import tempfile
from typing import Type, List, Dict, Optional, Any

from openai import OpenAI
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

# Streamlit Cloud için sqlite3 uyumluluğu
try:
    __import__("pysqlite3")          
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  
except Exception:
    pass

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import docx2txt
import pymupdf4llm

try:
    # pydantic v2
    from pydantic import BaseModel, Field, ConfigDict
    PYD_VER = 2
except Exception:  # pydantic v1 fallback
    from pydantic import BaseModel, Field  # type: ignore
    ConfigDict = None  # type: ignore
    PYD_VER = 1


# =========================
# Agentic RAG Tool Şemaları
# =========================
class RewriteInput(BaseModel):
    question: str = Field(..., description="Kullanıcının orijinal sorusu")
    n: int = Field(3, description="Kaç yeniden yazım üretilecek (1-5 arası önerilir)")


class QueryRewriteTool(BaseTool):
    name: str = "query_rewrite_tool"
    description: str = "Kullanıcının sorusunun bilgi-getirmeyi kolaylaştıracak semantik varyasyonlarını LLM ile üretir."
    args_schema: Type[BaseModel] = RewriteInput
    openai_api_key: str = ""

    def _run(self, question: str, n: int = 3) -> str:
        if not self.openai_api_key:
            return json.dumps([question, question + " hakkında detay"])
        
        client = OpenAI(api_key=self.openai_api_key)
        prompt = (
            f"Kullanıcının şu sorusu için RAG vektör veritabanında arama yapmak üzere {n} farklı "
            f"semantik olarak zenginleştirilmiş varyasyon üret. Yalnızca sonuçları barındıran bir "
            f"JSON listesi (array) döndür. Soru: {question}"
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            output = response.choices[0].message.content.strip()
            
            # Markdown code block temizliği
            if output.startswith("```json"):
                output = output[7:-3]
            elif output.startswith("```"):
                output = output[3:-3]
                
            queries = json.loads(output)
            if isinstance(queries, list):
                return json.dumps(queries, ensure_ascii=False)
        except Exception:
            pass # Hata durumunda fallback
            
        return json.dumps([question, question + " detaylı"], ensure_ascii=False)


class RetrieverMixInput(BaseModel):
    queries: List[str] = Field(..., description="Arama için sorgular")
    k: int = Field(4, description="Her sorgu için kaç sonuç")


class RetrieverMixTool(BaseTool):
    name: str = "retriever_mix_tool"
    description: str = "Çoklu sorguyu kullanarak FAISS üzerinden sonuçları toplar ve tek bir bağlam döndürür."
    args_schema: Type[BaseModel] = RetrieverMixInput

    if PYD_VER == 2:
        model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')  # type: ignore
    else:  # v1
        class Config:
            arbitrary_types_allowed = True
            extra = "allow"

    vectordb: Optional[Any] = None

    def _run(self, queries: List[str], k: int = 4) -> str:
        if self.vectordb is None:
            return json.dumps({"error": "vectordb_not_ready"})

        chunks: Dict[str, str] = {}
        for q in queries:
            docs = self.vectordb.similarity_search(q, k=k)
            for d in docs:
                source = d.metadata.get("source", "unknown")
                if source not in chunks:
                    chunks[source] = d.page_content

        context_sections = [text.strip() for text in chunks.values() if text.strip()]
        context = "\n\n---\n\n".join(context_sections) if context_sections else ""
        return json.dumps({"context": context}, ensure_ascii=False)


# =========================
# ChatBot Sınıfı
# =========================
class ChatBot:
    def __init__(self):
        self.temperature = 0.5
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.top_p = 1.0

        self.openai_api_key = None
        self.openai_client = None
        self.embeddings = None
        self.memory = None
        self.vectordb: Optional[FAISS] = None
        self.selected_api = None
        self.language = "English"
        self.selected_model = None
        self.is_uploaded = False
        self.instruction = "You are a helpful AI assistant."
        self.document_content = ""

        self.crewai_ready = False

        self.document_summary_agent = None
        self.document_analyzer_agent = None

        self.query_rewrite_tool = QueryRewriteTool()
        self.retriever_mix_tool = RetrieverMixTool(vectordb=None)

        self.planner_router_agent = None
        self.query_rewriter_agent = None
        self.retriever_mixer_agent = None

        self.chain_type_kwargs = {"prompt": self._create_prompt_template("English", self.instruction)}

    def set_prompt_instruction(self, instruction):
        self.instruction = instruction
        self._create_prompt_template(self.language, instruction)

    def _create_prompt_template(self, language, instruction):
        if language == "Turkish":
            prompt_text = """
            FileWiseGPT - Eşsiz Bilgi Asistanınız

            Sohbet Geçmişi:
            {chat_history}

            Dosya İçeriği:
            {context}

            Kullanıcının Sorusu:
            {question}

            Yanıtınızı Türkçe olarak verin. {addition}
            """
        else:
            prompt_text = """
            FileWiseGPT - Your Unrivaled Knowledge Assistant

            Chat History:
            {chat_history}

            File Context:
            {context}

            User's Question:
            {question}

            Answer in English. {addition}
            """
        prompt_text = prompt_text.replace("{addition}", instruction)
        PROMPT = PromptTemplate(template=prompt_text, input_variables=["context", "chat_history", "question"])
        self.chain_type_kwargs = {"prompt": PROMPT}
        return PROMPT

    def set_openai_api_key(self, key):
        self.openai_api_key = key
        self.openai_client = OpenAI(api_key=key)
        self.query_rewrite_tool.openai_api_key = key
        os.environ['OPENAI_API_KEY'] = key
        if self.memory is None:
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._initialize_crewai()

    def _initialize_crewai(self):
        if not self.openai_api_key:
            self.crewai_ready = False
            return

        # Statik tool'ları kaldırdık. Artık LLM kendi zekasıyla metni analiz ediyor.
        self.document_summary_agent = Agent(
            role='Document Summarizer',
            goal='Create comprehensive summaries of uploaded documents based strictly on context',
            backstory="You are an expert document summarizer capable of synthesizing large texts.",
            tools=[], 
            verbose=True,
            allow_delegation=False,
            llm_config={"model": self.selected_model or "gpt-4o-mini", "temperature": self.temperature}
        )

        self.document_analyzer_agent = Agent(
            role='Document Analyzer',
            goal='Analyze documents for insights, patterns, and key information',
            backstory="You are a seasoned document analyst.",
            tools=[],
            verbose=True,
            allow_delegation=False,
            llm_config={"model": self.selected_model or "gpt-4o-mini", "temperature": self.temperature}
        )

        self.planner_router_agent = Agent(
            role="Planner/Router",
            goal="Kullanıcı sorusuna en uygun çözüm akışını planla ve RAG gereksinimini belirle.",
            backstory="Deneyimli bir çözüm mimarısın.",
            tools=[],
            verbose=True,
            allow_delegation=False,
            llm_config={"model": self.selected_model or "gpt-4o-mini", "temperature": 0.1}
        )

        self.query_rewriter_agent = Agent(
            role="Query Rewriter",
            goal="Sorguyu bilgi erişimi için optimize eden yeniden yazımlar üret.",
            backstory="Arama sorgusu mühendisliğinde uzmansın.",
            tools=[self.query_rewrite_tool],
            verbose=True,
            allow_delegation=False,
            llm_config={"model": self.selected_model or "gpt-4o-mini", "temperature": 0.3}
        )

        self.retriever_mix_tool.vectordb = self.vectordb
        self.retriever_mixer_agent = Agent(
            role="Retriever Mixer",
            goal="Çoklu sorgu ile en iyi parçaları getir ve tek bir bağlam üret.",
            backstory="Büyük arşivlerden en alakalı bilgiyi harmanlamada uzmansın.",
            tools=[self.retriever_mix_tool],
            verbose=True,
            allow_delegation=False,
            llm_config={"model": self.selected_model or "gpt-4o-mini", "temperature": 0.1}
        )

        self.crewai_ready = True

    def set_temperature(self, temperature):
        self.temperature = temperature
        if self.crewai_ready:
            self._initialize_crewai()

    def set_presence_penalty(self, presence_penalty): self.presence_penalty = presence_penalty
    def set_frequency_penalty(self, frequency_penalty): self.frequency_penalty = frequency_penalty
    def set_top_p(self, top_p): self.top_p = top_p
    def set_language(self, language):
        self.language = language
        self.chain_type_kwargs = {"prompt": self._create_prompt_template(language, self.instruction)}
    def get_selected_language(self): return str(self.language)
    def select_api(self, api_name): self.selected_api = api_name
    def select_model(self): return ["gpt-4o-mini", "gpt-4o"] if self.selected_api == "OpenAI" else []
    def set_selected_model(self, selected_model):
        self.selected_model = selected_model
        if self.crewai_ready: self._initialize_crewai()

    # --------- Veri Okuma Katmanı (pymupdf4llm entegrasyonu) ---------
    def upload_file(self, uploaded_files):
        text = ""
        for file in uploaded_files:
            if file.type == "text/plain":
                text += str(file.read(), "utf-8") + "\n"
            elif file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                try:
                    # PDF to Markdown ile formatı koruma
                    md_text = pymupdf4llm.to_markdown(tmp_path)
                    text += md_text + "\n\n"
                except Exception as e:
                    st.error(f"PDF işlenirken hata: {e}")
                finally:
                    os.remove(tmp_path)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text += (docx2txt.process(file) or "") + "\n"
            else:
                st.error('Desteklenmeyen dosya formatı!', icon="🚨")

        self.document_content = text.strip()
        if self.document_content:
            self._text_to_chunks(self.document_content)
            self.is_uploaded = True
            if self.retriever_mix_tool:
                self.retriever_mix_tool.vectordb = self.vectordb
            if self.openai_api_key:
                self._initialize_crewai()

    # --------- Gelişmiş Chunking ---------
    def _text_to_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        self._chunks_to_vdb(chunks)

    def _chunks_to_vdb(self, text_chunks: List[str]):
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        metadatas = [{"source": f"chunk-{i}"} for i in range(len(text_chunks))]
        self.vectordb = FAISS.from_texts(text_chunks, embeddings, metadatas=metadatas)

    # --------- CrewAI Çalıştırıcıları (LLM Destekli Dinamik) ---------
    def get_document_summary(self):
        if not self.crewai_ready or not self.document_content: return "Belge yüklenmedi veya CrewAI hazır değil."
        summary_task = Task(
            description=f"Aşağıdaki belge içeriğini inceleyerek, yapısal bütünlüğü koruyan kapsamlı ama net maddelerle özetle:\n\n{self.document_content[:15000]}",
            agent=self.document_summary_agent,
            expected_output="Geniş bağlamı kapsayan profesyonel bir metin özeti."
        )
        crew = Crew(agents=[self.document_summary_agent], tasks=[summary_task], process=Process.sequential)
        return str(crew.kickoff())

    def get_document_analysis(self):
        if not self.crewai_ready or not self.document_content: return "Belge yüklenmedi veya CrewAI hazır değil."
        analysis_task = Task(
            description=f"Aşağıdaki belgeyi derinden analiz et; ana temaları, teknik veya önemli anahtar kelimeleri ve asıl vurgulanmak istenen içgörüleri ortaya çıkar:\n\n{self.document_content[:15000]}",
            agent=self.document_analyzer_agent,
            expected_output="Analitik, başlıklarla ayrılmış derinlemesine belge incelemesi."
        )
        crew = Crew(agents=[self.document_analyzer_agent], tasks=[analysis_task], process=Process.sequential)
        return str(crew.kickoff())

    def get_coordinated_analysis(self):
        if not self.crewai_ready or not self.document_content: return "Belge yüklenmedi veya CrewAI hazır değil."
        summary_task = Task(
            description=f"Belgeyi özetle:\n\n{self.document_content[:10000]}",
            agent=self.document_summary_agent,
            expected_output="Temel özet."
        )
        analysis_task = Task(
            description="Özeti okuyarak içerikteki en kritik içgörü ve trendleri birleştir.",
            agent=self.document_analyzer_agent,
            expected_output="Özet üzerinden genişletilmiş koordineli analiz."
        )
        crew = Crew(agents=[self.document_summary_agent, self.document_analyzer_agent], tasks=[summary_task, analysis_task], process=Process.sequential)
        return str(crew.kickoff())

    # --------- Agentic RAG ---------
    def ask_with_rag_agents(self, user_input: str) -> str:
        if not self.crewai_ready: return "CrewAI hazır değil."
        if self.vectordb is None: return "Önce belge yükleyin."

        plan_task = Task(
            description=f"Kullanıcı sorusu: '{user_input}'. Çözüm için Agentic RAG mimarisinin bilgi getirme akışını planla.",
            agent=self.planner_router_agent, expected_output="Kısa plan."
        )
        rewrite_task = Task(
            description="Soru için 3 yeniden yazım üret. 'query_rewrite_tool' kullan ve JSON liste döndür.",
            agent=self.query_rewriter_agent, expected_output="JSON listesi.", context=[plan_task]
        )
        retriever_task = Task(
            description="Yukarıdaki JSON listeyi baz alarak 'retriever_mix_tool' ile bağlam oluştur ve sadece bağlam metnini döndür.",
            agent=self.retriever_mixer_agent, expected_output='Arama bağlamı JSON', context=[rewrite_task]
        )
        answer_task = Task(
            description=f"Kullanıcının sorusu: {user_input}\nBulunan 'context' üzerinden akıcı, Markdown uyumlu ve profesyonel bir cevap yaz. Cevabı kurgulama, sadece bilgiye dayan.",
            agent=self.planner_router_agent, expected_output="Nihai cevap.", context=[retriever_task]
        )

        crew = Crew(
            agents=[self.planner_router_agent, self.query_rewriter_agent, self.retriever_mixer_agent],
            tasks=[plan_task, rewrite_task, retriever_task, answer_task],
            process=Process.sequential,
            memory=True, # CrewAI hafıza aktifleştirildi
            verbose=True
        )
        return f"### Yanıt\n\n{str(crew.kickoff()).strip()}"

    # --------- Standart RAG (Akış / Streaming Özelliği Eklendi) ---------
    def _query_stream(self, user_input):
        docs = self.vectordb.similarity_search(user_input) if self.vectordb else []
        context = "\n".join([doc.page_content for doc in docs]) if docs else self.document_content[:2000]

        prompt_template = self.chain_type_kwargs['prompt'].template
        prompt = prompt_template.format(context=context, chat_history='', question=user_input)

        response = self.openai_client.chat.completions.create(
            model=self.selected_model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            top_p=self.top_p,
            stream=True # Akış etkinleştirildi
        )

        full_answer = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                text_chunk = chunk.choices[0].delta.content
                full_answer += text_chunk
                yield text_chunk # Streamlit için yield

        if self.memory:
            self.memory.save_context({"input": user_input}, {"output": full_answer})