import streamlit as st
from ChatBot import ChatBot
import random
import os

class ChatBotInterface:
    def __init__(self):
        # ChatBot örneğini kalıcı hale getir
        if "chatbot" not in st.session_state:
            st.session_state["chatbot"] = ChatBot()
        self.chatbot: ChatBot = st.session_state["chatbot"]

        if "history" not in st.session_state:
            st.session_state["history"] = []

        if "use_agentic_rag" not in st.session_state:
            st.session_state["use_agentic_rag"] = True  # varsayılan: Agentic RAG açık

        self.wait_arr_tr = [
            "Bunu biliyor muydunuz? Penguenlerin bir kralı vardır! 2008'de Norveç bir pengueni 'Sir Nils Olav' olarak şövalye ilan etti.",
            "Bunu biliyor muydunuz? Domates aslında bir meyvedir, ancak 1893'te ABD Yüksek Mahkemesi onu sebze olarak sınıflandırdı!",
            "Bunu biliyor muydunuz? Dünyanın en kısa savaşı 1896'da İngiltere ve Zanzibar arasında oldu ve sadece 38 dakika sürdü.",
            "Bunu biliyor muydunuz? Ay'da 'arazi' satın alabilirsiniz! Resmi olmasa da bazı şirketler böyle satış yapıyor.",
            "Bunu biliyor muydunuz? Dünyanın en uzun yer ismi Galler'de ve 58 harften oluşuyor.",
        ]
        self.wait_arr_eng = [
            "Did you know? Penguins have a king! In 2008, Norway knighted a penguin named 'Sir Nils Olav'.",
            "Did you know? Tomatoes are a fruit, but in 1893 the U.S. Supreme Court classified them as a vegetable!",
            "Did you know? The shortest war was England vs. Zanzibar in 1896, lasting only 38 minutes.",
            "Did you know? You can buy 'land' on the Moon (not legally recognized).",
            "Did you know? The world's longest place name is in Wales with 58 letters.",
        ]

    def get_text(self):
        return st.chat_input("You: ", key="input")

    def run(self):
        st.title("FileWiseGPT 🤖 Enhanced with Agentic RAG")
        st.info("🚀 Agents integrated. You can upload documents and use the agent buttons, or run the chat with 'Agentic RAG'.")

        # Prompt alanı
        instruction = st.text_area("Your prompt", value=self.chatbot.instruction)
        if instruction != self.chatbot.instruction:
            self.chatbot.set_prompt_instruction(instruction)

        # Hoş geldin mesajını ilk kez göster
        if len(st.session_state["history"]) == 0:
            self.display_language_message()

        # ÖNCE sidebar işlemleri (durum güncellensin)
        self.sidebar_operations()

        # --- CrewAI Analiz Bölümü (sidebar’dan SONRA çiz) ---
        if self.chatbot.is_uploaded and self.chatbot.openai_api_key and self.chatbot.crewai_ready:
            st.markdown("---")
            st.subheader("🧠 Agent Analysis Tools")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📄 Generate Summary", help="Use AI agent to summarize the document"):
                    with st.spinner("Agents Summary Agent is running..."):
                        summary = self.chatbot.get_document_summary()
                        st.session_state["history"].append({"role": "assistant", "content": f"**Document Summary:**\n\n{summary}"})
                        st.success("Summary ready.")

            with col2:
                if st.button("🔎 Analyze Document", help="Use AI agent to analyze the document"):
                    with st.spinner("Analyzer Agent is running..."):
                        analysis = self.chatbot.get_document_analysis()
                        st.session_state["history"].append({"role": "assistant", "content": f"**Document Analysis:**\n\n{analysis}"})
                        st.success("Analysis ready.")

            with col3:
                if st.button("🎯 Coordinated Analysis", help="Use both agents working together"):
                    with st.spinner("Agents are coordinating..."):
                        coordinated = self.chatbot.get_coordinated_analysis()
                        st.session_state["history"].append({"role": "assistant", "content": f"**Coordinated Analysis:**\n\n{coordinated}"})
                        st.success("Coordinated analysis ready.")
            st.markdown("---")

        # Sohbet geçmişini yazdır
        for message in st.session_state["history"]:
            if message["role"] == "user":
                with st.chat_message(message["role"], avatar="https://i.hizliresim.com/f37txtv.png"):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"], avatar="https://i.hizliresim.com/n38vi8v.png"):
                    st.markdown(message["content"])

        # Chat girişi (belge yüklendiyse aktif)
        if self.chatbot.is_uploaded:
            user_input = self.get_text()
            if user_input:
                st.chat_message("user", avatar="https://i.hizliresim.com/f37txtv.png").markdown(user_input)
                st.session_state["history"].append({"role": "user", "content": user_input})
                random_message = random.choice(self.wait_arr_eng if self.chatbot.get_selected_language() == 'English' else self.wait_arr_tr)
                with st.spinner(random_message):
                    if st.session_state["use_agentic_rag"]:
                        output = self.chatbot.ask_with_rag_agents(user_input)
                    else:
                        output = self.chatbot._query(user_input)
                    with st.chat_message("assistant", avatar="https://i.hizliresim.com/n38vi8v.png"):
                        st.markdown(output)
                    st.session_state["history"].append({"role": "assistant", "content": output})

    def sidebar_operations(self):
        with st.sidebar:
            st.image("https://i.hizliresim.com/nitixu2.png", width=250)
            st.markdown("### 🤖 Agents Status")

            # Dil seçimi
            selected_language = "English"
            if selected_language != self.chatbot.language:
                self.chatbot.set_language(selected_language)

            # API ve model seçimleri
            selected_api = 'OpenAI'
            self.chatbot.select_api(selected_api)
            models = self.chatbot.select_model()
            if models:
                default_idx = models.index(self.chatbot.selected_model) if self.chatbot.selected_model in models else 0
                selected_model = st.selectbox('Select a model', models, index=default_idx)
                if selected_model != self.chatbot.selected_model:
                    self.chatbot.set_selected_model(selected_model)

            # Dosya yükleme
            uploaded_files = st.file_uploader("Choose files 📂", accept_multiple_files=True)
            if uploaded_files is not None and len(uploaded_files) > 0:
                with st.spinner("Processing documents..."):
                    self.chatbot.upload_file(uploaded_files)
                st.success("Documents uploaded successfully!")

            # API Key
            openai_api_key = st.text_input("OpenAI API Key", type="password", value=self.chatbot.openai_api_key or "")
            if openai_api_key and openai_api_key != self.chatbot.openai_api_key:
                self.chatbot.set_openai_api_key(openai_api_key)
                os.environ['OPENAI_API_KEY'] = openai_api_key
                st.success("API Key set and Agents initialized!")

            # CrewAI gösterimi
            if self.chatbot.openai_api_key and self.chatbot.crewai_ready:
                st.success("Agents Initialized ✅")
                st.write("Agents: Document Summarizer, Document Analyzer, Planner/Router, Query Rewriter, Retriever Mixer, Citation Enforcer, RAG Evaluator")
            else:
                st.warning("Agents Not Ready ⚠️")
                st.write("Please add OpenAI API Key")

            st.markdown("---")
            st.markdown("### 🧩 Agentic Chat Mode")
            use_agentic = st.toggle("Use Agentic RAG (Planner→Rewriter→Retriever→Citations→Evaluator)", value=st.session_state["use_agentic_rag"])
            if use_agentic != st.session_state["use_agentic_rag"]:
                st.session_state["use_agentic_rag"] = use_agentic

            st.markdown("---")
            st.markdown("### ⚙️ Model Parameters")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.01, value=self.chatbot.temperature)
            if temperature != self.chatbot.temperature:
                self.chatbot.set_temperature(temperature)

            presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, step=0.01, value=self.chatbot.presence_penalty)
            if presence_penalty != self.chatbot.presence_penalty:
                self.chatbot.set_presence_penalty(presence_penalty)

            freq_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, step=0.01, value=self.chatbot.frequency_penalty)
            if freq_penalty != self.chatbot.frequency_penalty:
                self.chatbot.set_frequency_penalty(freq_penalty)

            top_p = st.slider("Top_P", min_value=0.0, max_value=1.0, step=0.01, value=self.chatbot.top_p)
            if top_p != self.chatbot.top_p:
                self.chatbot.set_top_p(top_p)

            with st.expander("See explanation"):
                st.write("**Temperature:** Controls randomness. High = creative, low = focused.")
                st.write("**Presence Penalty:** Discourages repeating the same topics.")
                st.write("**Frequency Penalty:** Discourages repeating the same words.")
                st.write("**Top P:** Limits choices to the most likely words (e.g., top 90%).")

            with st.expander("🤖 About Agentic RAG Integration"):
                st.write("""
                **Agents:**
                - Document Summarizer
                - Document Analyzer
                - Planner/Router
                - Query Rewriter
                - Retriever Mixer
                - RAG Evaluator

                **Features:**
                - Sequential processing
                - Specialized tools
                - Multi-agent RAG pipeline
                """)

    def display_language_message(self):
        if self.chatbot.get_selected_language() == 'English':
            message = "Hi! Welcome to FileWiseGPT enhanced with Agentic RAG! 🚀\n\nUpload a file and add your OpenAI API key to start."
        else:
            message = ""
        st.session_state["history"].append({"role": "assistant", "content": message})
