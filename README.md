# FileWiseGPT 🤖 - Enhanced with Agentic RAG

FileWiseGPT is an advanced, AI-powered document analysis and chat assistant. Upgraded with an **Agentic Retrieval-Augmented Generation (RAG)** architecture using CrewAI and modern LangChain, it goes beyond simple vector searches. It intelligently plans, rewrites queries, retrieves context, and synthesizes highly accurate answers from your documents.

## Key Features

* **Agentic RAG Architecture:** Employs a collaborative team of AI agents to handle user queries:
    * **Planner/Router:** Analyzes the user's request and plans the retrieval strategy.
    * **Query Rewriter:** Uses LLM capabilities to semantically expand and optimize the search query.
    * **Retriever Mixer:** Executes multi-query searches on the FAISS vector database and aggregates the best context.
* **Advanced Document Ingestion:** Replaced legacy PDF readers with `pymupdf4llm` to preserve document structures, tables, and convert them directly into Markdown for maximum LLM comprehension.
* **Modern & Seamless UI:** Built with Streamlit 1.37+ utilizing `@st.fragment`. Sidebar operations (like parameter tuning or API key inputs) run independently, preventing full-page reloads and resetting of the chat interface.
* **Real-Time Streaming:** Enjoy token-by-token text streaming for a natural, fast chat experience.
* **Dedicated Analysis Agents:** Generate one-click comprehensive summaries, deep thematic analyses, or coordinated multi-agent insights directly from the sidebar.
* **Native Session Memory:** Ditched clunky legacy memory chains for a clean, Streamlit native session-state memory injection, reducing overhead and improving context awareness.

## Tech Stack

* **UI Framework:** [Streamlit](https://streamlit.io/)
* **Agent Orchestration:** [CrewAI](https://www.crewai.com/)
* **LLM Framework:** Modular [LangChain](https://www.langchain.com/) (Core, Text Splitters, Community)
* **Vector Database:** FAISS
* **Document Parsing:** PyMuPDF4LLM, docx2txt
* **Embeddings & Models:** OpenAI API (`gpt-4o-mini`, `gpt-4o`)

## Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/FileWiseGPT.git](https://github.com/yourusername/FileWiseGPT.git)
cd FileWiseGPT
```

2. **Install the dependencies:**
Make sure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

3. **Environment Setup (Optional):**
You can create a `.env` file in the root directory to store your API keys permanently, or just input them directly via the UI.

## Usage

1. **Start the Streamlit app:**
```bash
streamlit run src/app.py
```

2. **Interact via the UI:**
* Open the provided local URL in your browser.
* Enter your **OpenAI API Key** in the sidebar.
* Upload your documents (`.pdf`, `.docx`, `.txt`).
* Choose between standard Streaming RAG or toggle **"Use Agentic RAG"** for complex, multi-step reasoning.
* Use the quick-action agent buttons to generate instant summaries or deep document analyses.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.