# FileWiseGPT 🤖 Enhanced with Agentic RAG

An advanced AI-powered document analysis chatbot that leverages multiple specialized agents for intelligent document processing and question answering. Built with OpenAI GPT models, CrewAI agent framework, and enhanced RAG (Retrieval-Augmented Generation) capabilities.

## Features

### Core Capabilities
- **Multi-Format Document Support**: Process PDF, DOCX, and TXT files seamlessly
- **Agentic RAG Architecture**: Advanced multi-agent system for intelligent document retrieval and analysis
- **AI-Powered Responses**: Utilizes OpenAI GPT models (GPT-4o, GPT-4o-mini) for high-quality answers
- **Specialized Agent System**: Multiple AI agents working in coordination for optimal results
- **Vector Database Integration**: FAISS-based similarity search for precise document retrieval
- **Customizable Parameters**: Fine-tune model behavior with temperature, penalties, and other parameters
- **Streamlit Interface**: Clean, intuitive web interface for seamless interaction

### Agentic RAG System
The enhanced FileWiseGPT employs a sophisticated multi-agent architecture:

- **Document Summarizer Agent**: Creates comprehensive document summaries
- **Document Analyzer Agent**: Extracts insights, patterns, and key information
- **Planner/Router Agent**: Orchestrates query processing workflow
- **Query Rewriter Agent**: Optimizes queries for better information retrieval
- **Retriever Mixer Agent**: Combines multiple query results for comprehensive context

### Intelligent Processing Modes
- **Agentic RAG Mode**: Full multi-agent pipeline with query rewriting, retrieval mixing, and coordinated analysis
- **Standard RAG Mode**: Traditional retrieval-augmented generation for direct answers
- **Agent Analysis Tools**: One-click document summary, analysis, and coordinated insights

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/agining/FileWiseGPT.git
cd FileWiseGPT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
cd src
streamlit run app.py
```

## Usage

### Initial Setup
1. Launch the application and navigate to the sidebar
2. Enter your OpenAI API key
3. Upload your documents (PDF, DOCX, or TXT)
4. Wait for document processing and agent initialization

![Sidebar Usage](https://i.imgur.com/zd6WeLY.gif)

### Agent Analysis Tools
Once documents are uploaded and agents are initialized, use the specialized analysis tools:

- **Generate Summary**: AI agent creates a structured summary of your document
- **Analyze Document**: Deep analysis revealing themes, keywords, and insights  
- **Coordinated Analysis**: Multiple agents working together for comprehensive understanding

![Summary and Analysis Example](https://i.imgur.com/3w08smS.gifv)

### Interactive Chat Modes

#### Agentic RAG Mode (Recommended)
Enable "Agentic RAG" for the most sophisticated query processing:
- Query gets rewritten and optimized by specialized agents
- Multiple retrieval strategies ensure comprehensive context
- Coordinated multi-agent response generation

![Agentic RAG Query Example](https://i.imgur.com/fl8YJnM.gifv)

#### Standard RAG Mode  
Disable "Agentic RAG" for traditional retrieval-based responses:
- Direct similarity search and response generation
- Faster processing for simple queries
- Standard OpenAI model responses

![Standard RAG Query Example](https://i.imgur.com/H3H2Mzz.gif)

## Configuration

### Model Parameters
Fine-tune the AI behavior through the sidebar controls:

- **Temperature** (0.0-1.0): Controls response creativity and randomness
- **Presence Penalty** (-2.0-2.0): Reduces topic repetition
- **Frequency Penalty** (-2.0-2.0): Reduces word repetition  
- **Top P** (0.0-1.0): Limits vocabulary to most probable words

### Custom Prompts
Modify the system prompt to customize the AI assistant's behavior and response style for specific use cases.

## Technical Architecture

### Dependencies
- **Streamlit**: Web interface framework
- **OpenAI**: GPT model integration
- **LangChain**: Document processing and RAG pipeline
- **CrewAI**: Multi-agent coordination framework
- **FAISS**: Vector similarity search
- **PyPDF2**: PDF document processing
- **docx2txt**: Word document processing

### Agent Workflow
1. **Planning Phase**: Planner/Router agent analyzes the query and determines optimal processing strategy
2. **Query Enhancement**: Query Rewriter generates multiple optimized query variations
3. **Information Retrieval**: Retriever Mixer combines results from multiple query strategies
4. **Response Generation**: Coordinated agents generate comprehensive, contextually-aware responses

## Supported File Formats

- **PDF**: Full text extraction with formatting preservation
- **DOCX**: Microsoft Word document processing
- **TXT**: Plain text file support

## Online Demo

[Check out the live version here](https://filewisegpt.streamlit.app/)

## License

[MIT](https://github.com/agining/FileWiseGPT/blob/main/LICENSE)

## Acknowledgments

This project integrates several powerful AI and ML libraries to deliver advanced document analysis capabilities. Special recognition to the CrewAI framework for enabling sophisticated multi-agent coordination.