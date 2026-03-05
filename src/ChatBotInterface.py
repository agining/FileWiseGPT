import streamlit as st
from ChatBot import ChatBot
import random
import os

class ChatBotInterface:
    def __init__(self):
        if "chatbot" not in st.session_state:
            st.session_state["chatbot"] = ChatBot()
        self.chatbot: ChatBot = st.session_state["chatbot"]

        if "history" not in st.session_state:
            st.session_state["history"] = []

        if "use_agentic_rag" not in st.session_state:
            st.session_state["use_agentic_rag"] = True

    def get_text(self):
        return st.chat_input("Mesajınızı yazın...", key="input")

    def run(self):
        st.title("FileWiseGPT 🤖 Enhanced with Agentic RAG")
        st.info("🚀 Agents integrated. You can upload documents and use the agent buttons, or run the chat with 'Agentic RAG'.")

        instruction = st.text_area("Your prompt", value=self.chatbot.instruction)
        if instruction != self.chatbot.instruction:
            self.chatbot.set_prompt_instruction(instruction)

        if len(st.session_state["history"]) == 0:
            self.display_language_message()

        # Render sidebar as a fragment (Sayfa yenilenmesini engeller)
        self.sidebar_operations()

        if self.chatbot.is_uploaded and self.chatbot.openai_api_key and self.chatbot.crewai_ready:
            st.markdown("---")
            st.subheader("🧠 Agent Analysis Tools")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📄 Generate Summary", help="Belgeyi LLM ile analiz ederek özetle"):
                    with st.spinner("Summary Agent is running..."):
                        summary = self.chatbot.get_document_summary()
                        st.session_state["history"].append({"role": "assistant", "content": f"**Document Summary:**\n\n{summary}"})
                        st.success("Summary ready.")

            with col2:
                if st.button("🔎 Analyze Document", help="Belgeyi derinlemesine analiz et"):
                    with st.spinner("Analyzer Agent is running..."):
                        analysis = self.chatbot.get_document_analysis()
                        st.session_state["history"].append({"role": "assistant", "content": f"**Document Analysis:**\n\n{analysis}"})
                        st.success("Analysis ready.")

            with col3:
                if st.button("🎯 Coordinated Analysis", help="İki ajan bir arada çalışır"):
                    with st.spinner("Agents are coordinating..."):
                        coordinated = self.chatbot.get_coordinated_analysis()
                        st.session_state["history"].append({"role": "assistant", "content": f"**Coordinated Analysis:**\n\n{coordinated}"})
                        st.success("Coordinated analysis ready.")
            st.markdown("---")

        for message in st.session_state["history"]:
            avatar = "https://i.hizliresim.com/f37txtv.png" if message["role"] == "user" else "https://i.hizliresim.com/n38vi8v.png"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if self.chatbot.is_uploaded:
            user_input = self.get_text()
            if user_input:
                st.chat_message("user", avatar="https://i.hizliresim.com/f37txtv.png").markdown(user_input)
                st.session_state["history"].append({"role": "user", "content": user_input})
                
                if st.session_state["use_agentic_rag"]:
                    # Ajanların durumunu izleme bileşeni
                    with st.status("🧠 Agentic RAG devrede, ajanlar planlıyor ve çalışıyor...", expanded=True) as status:
                        st.write("🔍 Planner ve Rewriter görevleri yürütülüyor...")
                        output = self.chatbot.ask_with_rag_agents(user_input)
                        status.update(label="✅ Ajanlar yanıtı başarıyla tamamladı!", state="complete", expanded=False)
                        
                    with st.chat_message("assistant", avatar="https://i.hizliresim.com/n38vi8v.png"):
                        st.markdown(output)
                    st.session_state["history"].append({"role": "assistant", "content": output})
                else:
                    # Akışkan Stream (Kelime Kelime) RAG Yanıtı
                    with st.chat_message("assistant", avatar="https://i.hizliresim.com/n38vi8v.png"):
                        response_stream = self.chatbot._query_stream(user_input)
                        output = st.write_stream(response_stream)
                    st.session_state["history"].append({"role": "assistant", "content": output})

    @st.fragment
    def sidebar_operations(self):
        with st.sidebar:
            st.image("https://i.hizliresim.com/nitixu2.png", width=250)
            st.markdown("### 🤖 Agents Status")

            selected_language = "English"
            if selected_language != self.chatbot.language:
                self.chatbot.set_language(selected_language)

            self.chatbot.select_api('OpenAI')
            models = self.chatbot.select_model()
            if models:
                default_idx = models.index(self.chatbot.selected_model) if self.chatbot.selected_model in models else 0
                selected_model = st.selectbox('Select a model', models, index=default_idx)
                if selected_model != self.chatbot.selected_model:
                    self.chatbot.set_selected_model(selected_model)

            uploaded_files = st.file_uploader("Choose files 📂", accept_multiple_files=True)
            if uploaded_files is not None and len(uploaded_files) > 0:
                with st.spinner("Processing documents..."):
                    self.chatbot.upload_file(uploaded_files)
                st.success("Documents uploaded successfully!")

            openai_api_key = st.text_input("OpenAI API Key", type="password", value=self.chatbot.openai_api_key or "")
            if openai_api_key and openai_api_key != self.chatbot.openai_api_key:
                self.chatbot.set_openai_api_key(openai_api_key)
                st.success("API Key set and Agents initialized!")

            if self.chatbot.openai_api_key and self.chatbot.crewai_ready:
                st.success("Agents Initialized ✅")
            else:
                st.warning("Agents Not Ready ⚠️")

            st.markdown("---")
            st.markdown("### 🧩 Agentic Chat Mode")
            use_agentic = st.toggle("Use Agentic RAG", value=st.session_state["use_agentic_rag"])
            if use_agentic != st.session_state["use_agentic_rag"]:
                st.session_state["use_agentic_rag"] = use_agentic

            st.markdown("---")
            st.markdown("### ⚙️ Model Parameters")
            temperature = st.slider("Temperature", 0.0, 1.0, self.chatbot.temperature)
            if temperature != self.chatbot.temperature: self.chatbot.set_temperature(temperature)
            
            presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, self.chatbot.presence_penalty)
            if presence_penalty != self.chatbot.presence_penalty: self.chatbot.set_presence_penalty(presence_penalty)
            
            freq_penalty = st.slider("Frequency Penalty", -2.0, 2.0, self.chatbot.frequency_penalty)
            if freq_penalty != self.chatbot.frequency_penalty: self.chatbot.set_frequency_penalty(freq_penalty)

            top_p = st.slider("Top_P", 0.0, 1.0, self.chatbot.top_p)
            if top_p != self.chatbot.top_p: self.chatbot.set_top_p(top_p)

    def display_language_message(self):
        message = "Hi! Welcome to FileWiseGPT enhanced with Agentic RAG! 🚀\n\nUpload a file and add your OpenAI API key to start."
        st.session_state["history"].append({"role": "assistant", "content": message})