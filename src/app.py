import streamlit as st
from ChatBotInterface import ChatBotInterface
from dotenv import load_dotenv

load_dotenv()

def main():
    st.set_page_config(
        page_title="FileWiseGPT - Enhanced with Agentic RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header { text-align: center; color: #2E86AB; padding: 1rem 0; }
    .info-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .success-box { background-color: #d4edda; color: #155724; padding: 0.75rem; border-radius: 0.25rem; margin: 0.5rem 0; border: 1px solid #c3e6cb; }
    .warning-box { background-color: #fff3cd; color: #856404; padding: 0.75rem; border-radius: 0.25rem; margin: 0.5rem 0; border: 1px solid #ffeaa7; }
    </style>
    """, unsafe_allow_html=True)

    try:
        interface = ChatBotInterface()
        interface.run()
    except Exception as e:
        st.error(f"Uygulama başlatılırken hata oluştu: {str(e)}")
        st.markdown("""
        ### Kontrol Listesi
        1. `pip install -r requirements.txt`
        2. Geçerli OpenAI API anahtarı
        3. PDF/DOCX/TXT dosyası yükleyin
        4. Sorun sürerse uygulamayı yeniden başlatın
        """)

if __name__ == "__main__":
    main()