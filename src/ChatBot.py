import os
import re
import json
from typing import Type, List, Dict, Optional, Any

from openai import OpenAI
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
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
from PyPDF2 import PdfReader

try:
    # pydantic v2
    from pydantic import BaseModel, Field, ConfigDict
    PYD_VER = 2
except Exception:  # pydantic v1 fallback
    from pydantic import BaseModel, Field  # type: ignore
    ConfigDict = None  # type: ignore
    PYD_VER = 1


class DocumentInput(BaseModel):
    content: str = Field(description="Analiz edilecek belge içeriği")


class DocumentSummaryTool(BaseTool):
    name: str = "document_summary_tool"
    description: str = "Belge içeriğini maddeler halinde özetler"
    args_schema: Type[BaseModel] = DocumentInput

    def _run(self, content: str) -> str:
        lines = content.split('\n')
        key_points = []
        for line in lines:
            if line.strip() and len(line) > 50:
                key_points.append(line.strip())
        summary = "Document Summary:\n"
        for i, point in enumerate(key_points[:5], 1):
            summary += f"{i}. {point[:200]}...\n" if len(point) > 200 else f"{i}. {point}\n"
        if not key_points:
            summary += "1. Metin kısa veya temizlenemedi.\n"
        return summary


class DocumentAnalyzerTool(BaseTool):
    name: str = "document_analyzer_tool"
    description: str = "Belgedeki tema, anahtar kelime ve öngörüleri çıkarır"
    args_schema: Type[BaseModel] = DocumentInput

    def _run(self, content: str) -> str:
        word_count = len(content.split())
        char_count = len(content)
        common_words = {}
        for word in content.lower().split():
            cleaned_word = re.sub(r'[^\w]', '', word)
            if len(cleaned_word) > 3:
                common_words[cleaned_word] = common_words.get(cleaned_word, 0) + 1
        top_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis = (
            "Document Analysis:\n"
            f"- Word Count: {word_count}\n"
            f"- Character Count: {char_count}\n"
            f"- Top Keywords: {', '.join([w for w, _ in top_words])}\n"
            f"- Document Length: {'Long' if word_count > 1000 else 'Medium' if word_count > 500 else 'Short'}\n"
        )
        return analysis


# =========================
# Agentic RAG Tool Şemaları
# =========================
class RewriteInput(BaseModel):
    question: str = Field(..., description="Kullanıcının orijinal sorusu")
    n: int = Field(3, description="Kaç yeniden yazım üretilecek (1-5 arası önerilir)")


class QueryRewriteTool(BaseTool):
    name: str = "query_rewrite_tool"
    description: str = (
        "Kullanıcının sorusunun bilgi-getirmeyi kolaylaştıracak 2-5 farklı "
        "yeniden yazımını üretir ve JSON listesi döner."
    )
    args_schema: Type[BaseModel] = RewriteInput

    def _run(self, question: str, n: int = 3) -> str:
        base = question.strip()
        variants = set()
        variants.add(base)
        variants.add(base.lower())
        if not base.endswith("?"):
            variants.add(base + "?")
        variants.add(f"Please answer: {base}")
        variants.add(f"Rephrase: {base}")
        variants.add(f"Detailed question: {base}")
        out = list(variants)[: max(1, min(n, 5))]
        return json.dumps(out, ensure_ascii=False)


class RetrieverMixInput(BaseModel):
    queries: List[str] = Field(..., description="Arama için sorgular")
    k: int = Field(4, description="Her sorgu için kaç sonuç")


class RetrieverMixTool(BaseTool):
    name: str = "retriever_mix_tool"
    description: str = (
        "Verilen birden fazla sorgu için FAISS vektör veritabanından sonuçları "
        "toplar, birleştirir ve tek bir metin bağlamı döndürür."
    )
    args_schema: Type[BaseModel] = RetrieverMixInput

    # Pydantic model config
    if PYD_VER == 2:
        model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')  # type: ignore
    else:  # v1
        class Config:
            arbitrary_types_allowed = True
            extra = "allow"

    # FAISS referansı
    vectordb: Optional[Any] = None

    def _run(self, queries: List[str], k: int = 4) -> str:
        if self.vectordb is None:
            return json.dumps({"error": "vectordb_not_ready"})

        # Topla & benzersizleştir
        chunks: Dict[str, str] = {}  # source -> text
        for q in queries:
            docs = self.vectordb.similarity_search(q, k=k)
            for d in docs:
                source = d.metadata.get("source", "unknown")
                if source not in chunks:
                    chunks[source] = d.page_content

        # Bağlamı tek bir blokta birleştir
        context_sections = []
        for _, text in chunks.items():
            snippet = (text or "").strip()
            if snippet:
                context_sections.append(snippet)

        context = "\n\n---\n\n".join(context_sections) if context_sections else ""
        return json.dumps({"context": context}, ensure_ascii=False)


# =========================
# ChatBot
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

        # CrewAI hazır bayrağı ve ajanlar
        self.crewai_ready = False

        # Varsayılan araçlar (özet/analiz)
        self.document_summary_agent = None
        self.document_analyzer_agent = None
        self.summary_tool = DocumentSummaryTool()
        self.analyzer_tool = DocumentAnalyzerTool()

        # Agentic RAG araç & ajanları
        self.query_rewrite_tool = QueryRewriteTool()
        self.retriever_mix_tool = RetrieverMixTool(vectordb=None)  # vdb yüklendikten sonra güncellenecek

        self.planner_router_agent = None
        self.query_rewriter_agent = None
        self.retriever_mixer_agent = None

        self.chain_type_kwargs = {"prompt": self._create_prompt_template("English", self.instruction)}

    # --------- Prompt ---------
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
        PROMPT = PromptTemplate(
            template=prompt_text, input_variables=["context", "chat_history", "question"]
        )
        self.chain_type_kwargs = {"prompt": PROMPT}
        return PROMPT

    # --------- OpenAI Ayar ---------
    def set_openai_api_key(self, key):
        self.openai_api_key = key
        self.openai_client = OpenAI(api_key=key)
        os.environ['OPENAI_API_KEY'] = key
        if self.memory is None:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        self._initialize_crewai()

    # --------- CrewAI Hazırlığı (ajanları hazırla) ---------
    def _initialize_crewai(self):
        if not self.openai_api_key:
            self.crewai_ready = False
            return

        # Özet / Analiz ajanları
        self.document_summary_agent = Agent(
            role='Document Summarizer',
            goal='Create comprehensive summaries of uploaded documents',
            backstory="You are an expert document summarizer.",
            tools=[self.summary_tool],
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": self.selected_model or "gpt-4o-mini",
                "temperature": self.temperature
            }
        )

        self.document_analyzer_agent = Agent(
            role='Document Analyzer',
            goal='Analyze documents for insights, patterns, and key information',
            backstory="You are a seasoned document analyst.",
            tools=[self.analyzer_tool],
            verbose=True,
            allow_delegation=False,
            llm_config={
                "model": self.selected_model or "gpt-4o-mini",
                "temperature": self.temperature
            }
        )

        # Agentic RAG ajanları
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

        # retriever tool'a mevcut vdb'yi tak
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

    def set_presence_penalty(self, presence_penalty):
        self.presence_penalty = presence_penalty

    def set_frequency_penalty(self, frequency_penalty):
        self.frequency_penalty = frequency_penalty

    def set_top_p(self, top_p):
        self.top_p = top_p

    def set_language(self, language):
        self.language = language
        self.chain_type_kwargs = {"prompt": self._create_prompt_template(language, self.instruction)}

    def get_selected_language(self):
        return str(self.language)

    def select_api(self, api_name):
        self.selected_api = api_name

    def select_model(self):
        if self.selected_api == "OpenAI":
            return ["gpt-5-nano", "gpt-4o-mini"]
        else:
            return []

    def set_selected_model(self, selected_model):
        self.selected_model = selected_model
        if self.crewai_ready:
            self._initialize_crewai()

    # --------- Dosya İşleme ---------
    def upload_file(self, uploaded_files):
        text = ""
        for file in uploaded_files:
            if file.type == "text/plain":
                text += str(file.read(), "utf-8") + "\n"
            elif file.type == "application/pdf":
                pdf = PdfReader(file)
                for page in pdf.pages:
                    output = page.extract_text() or ""
                    output = re.sub(r"(\w+)-\n(\w+)", r"\1\2", output)
                    output = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", output.strip())
                    output = re.sub(r"\n\s*\n", "\n\n", output)
                    text += output + "\n"
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text += (docx2txt.process(file) or "") + "\n"
            else:
                st.error('Desteklenmeyen dosya formatı!', icon="🚨")

        self.document_content = text.strip()
        if self.document_content:
            self._text_to_chunks(self.document_content)
            self.is_uploaded = True
            # vdb güncellendi; retriever tool'a tak
            if self.retriever_mix_tool:
                self.retriever_mix_tool.vectordb = self.vectordb

            # API anahtarı varsa ajanları yeniden hazırla (araç güncellemesi için)
            if self.openai_api_key:
                self._initialize_crewai()

    def _text_to_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        self._chunks_to_vdb(chunks)

    def _chunks_to_vdb(self, text_chunks: List[str]):
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        # Atıflar için her parçaya kaynak etiketi — (artık görünür atıf yok, ama izlemek için faydalı)
        metadatas = [{"source": f"chunk-{i}"} for i in range(len(text_chunks))]
        self.vectordb = FAISS.from_texts(text_chunks, embeddings, metadatas=metadatas)

    # --------- CrewAI Çalıştırıcıları (Özet / Analiz) ---------
    def get_document_summary(self):
        if not self.crewai_ready or not self.document_content:
            return "Belge yüklenmedi veya CrewAI hazır değil."

        summary_task = Task(
            description=(
                "Aşağıdaki içeriği kısa ve net maddelerle özetle. "
                "Gerekirse 'document_summary_tool' kullan:\n\n"
                f"{self.document_content[:3000]}"
            ),
            agent=self.document_summary_agent,
            expected_output="En fazla 8 maddelik bir özet."
        )

        crew = Crew(
            agents=[self.document_summary_agent],
            tasks=[summary_task],
            process=Process.sequential,
            verbose=True
        )
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"Özet oluşturulamadı: {str(e)}"

    def get_document_analysis(self):
        if not self.crewai_ready or not self.document_content:
            return "Belge yüklenmedi veya CrewAI hazır değil."

        analysis_task = Task(
            description=(
                "Aşağıdaki içeriği analiz et; tema, anahtar kelime ve çıkarımlar ver. "
                "Gerekirse 'document_analyzer_tool' kullan:\n\n"
                f"{self.document_content[:3000]}"
            ),
            agent=self.document_analyzer_agent,
            expected_output="Kısa başlıklar ve maddelerle bir analiz."
        )

        crew = Crew(
            agents=[self.document_analyzer_agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True
        )
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"Analiz oluşturulamadı: {str(e)}"

    def get_coordinated_analysis(self):
        if not self.crewai_ready or not self.document_content:
            return "Belge yüklenmedi veya CrewAI hazır değil."

        summary_task = Task(
            description=(
                "Belgeyi özetle (kısa ve net maddeler). Gerekirse 'document_summary_tool' kullan:\n\n"
                f"{self.document_content[:1500]}"
            ),
            agent=self.document_summary_agent,
            expected_output="En fazla 6 maddelik özet."
        )
        analysis_task = Task(
            description=(
                "Yukarıdaki özete ve belge içeriğine dayanarak kısa bir analiz çıkar. "
                "Gerekirse 'document_analyzer_tool' kullan:\n\n"
                f"{self.document_content[:1500]}"
            ),
            agent=self.document_analyzer_agent,
            expected_output="Temalar, anahtar kelimeler ve içgörüler."
        )

        crew = Crew(
            agents=[self.document_summary_agent, self.document_analyzer_agent],
            tasks=[summary_task, analysis_task],
            process=Process.sequential,
            verbose=True
        )
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"Koordineli analiz hatası: {str(e)}"

    # --------- Agentic RAG Sorgu (Citations YOK, Markdown Çıkış) ---------
    def ask_with_rag_agents(self, user_input: str) -> str:
        if not self.crewai_ready:
            return "CrewAI hazır değil. Lütfen OpenAI API anahtarını ekleyin."
        if self.vectordb is None:
            return "Önce bir belge yükleyin ki RAG ajanları bağlamdan yanıt verebilsin."

        # 1) Planlama
        plan_task = Task(
            description=(
                "Kullanıcının sorusu için işlem adımlarını planla: "
                "(1) Planner/Router → (2) Query Rewriter → (3) Retriever Mixer → (4) Final Answer. "
                f"Kullanıcı sorusu: '{user_input}'. "
                "Kısa bir plan metni üret."
            ),
            agent=self.planner_router_agent,
            expected_output="Kısa plan."
        )

        # 2) Query Rewriter
        rewrite_task = Task(
            description="Soru için 3 yeniden yazım üret. 'query_rewrite_tool' kullan ve JSON liste döndür.",
            agent=self.query_rewriter_agent,
            expected_output="JSON listesi (ör: [\"...\",\"...\",\"...\"]).",
            context=[plan_task]
        )

        # 3) Retriever Mixer
        retriever_task = Task(
            description="Yukarıdaki JSON listeden bağlam oluştur. 'retriever_mix_tool' çağır ve yalnızca 'context' döndür.",
            agent=self.retriever_mixer_agent,
            expected_output='{"context": "..."} biçiminde JSON.',
            context=[rewrite_task]
        )

        # 4) Final Answer (Markdown formatında, citations yok)
        answer_task = Task(
            description=(
                f"Kullanıcının sorusu: {user_input}\n"
                "Aşağıdaki 'context' içeriğini temel alarak **okunaklı, Markdown uyumlu** bir cevap yaz:\n"
                "- Cevap **Türkçe** ise başlıkları `###` ile, maddeleri `-` ile ver.\n"
                "- Gereğinde alt başlıklar, numaralı listeler ve kalın vurgular kullan.\n"
                "- Gereksiz tekrar yapma, **kaynak atfı/citation ekleme**.\n"
                "- Girişte kısa bir özet, ardından maddelerle detay ver ve sonda kısa bir sonuç yaz.\n"
                "Sadece nihai cevabı döndür."
            ),
            agent=self.planner_router_agent,
            expected_output="Markdown formatında net nihai cevap.",
            context=[retriever_task]
        )

        crew = Crew(
            agents=[
                self.planner_router_agent,
                self.query_rewriter_agent,
                self.retriever_mixer_agent
            ],
            tasks=[plan_task, rewrite_task, retriever_task, answer_task],
            process=Process.sequential,
            verbose=True
        )

        try:
            result = str(crew.kickoff())
        except Exception as e:
            return f"Ajanlı RAG çalıştırma hatası: {str(e)}"

        # Markdown uyumlu döndür
        return f"### Yanıt\n\n{result.strip()}"

    # --------- Basit (ajanlı olmayan) RAG ---------
    def _query(self, user_input):
        docs = self.vectordb.similarity_search(user_input) if self.vectordb else []
        context = "\n".join([doc.page_content for doc in docs]) if docs else self.document_content[:2000]

        prompt_template = self.chain_type_kwargs['prompt'].template
        prompt = prompt_template.format(
            context=context,
            chat_history='',
            question=user_input
        )

        response = self.openai_client.chat.completions.create(
            model=self.selected_model or "gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            top_p=self.top_p
        )

        answer = response.choices[0].message.content
        if self.memory:
            self.memory.save_context({"input": user_input}, {"output": answer})
        return answer
