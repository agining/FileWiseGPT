import streamlit as st
from ChatBot import ChatBot
import time
import random
import toml
import os

class ChatBotInterface:
    def __init__(self):
        self.chatbot = ChatBot()
        self.time_arr = [0.012,0.061,0.023,0.032,0.04,0.08,0.073,0.09,0.025,0.012,0.01,0.03,0.016,0.06,0.05,0.03,0.08,0.063,0.09,0.08]
        self.wait_arr_tr = ["Eğer bir süper gücün olsaydı, hangisini seçerdin ve neden?","Bir zaman makinen olsa, hangi döneme gitmek isterdin?",
                            "Favori dondurma lezzetin nedir?","Bir hayvan olabilseydin hangisini tercih ederdin?",
                            "Hangi ünlüyle akşam yemeği yemek isterdin?","En son okuduğun kitap neydi ve bu kitap hakkında ne düşünüyorsun?",
                            "Bir ada tatiline çıkacak olsan, yanına alacağın üç şey ne olurdu?","Eğer bir film karakteri olabilseydin kim olurdun?",
                            "Bir pizzada hangi malzemeler olmazsa olmaz?","Çocukken ne olmak istiyordun ve şu anki mesleğinle nasıl bir ilişkisi var?"]
        self.wait_arr_eng = ["If you could have any superpower, what would it be and why?","If you had a time machine, which period would you visit?",
                             "What's your favorite ice cream flavor and why do you choose it?","If you could be any animal, which one would you choose and why?",
                             "Which celebrity would you like to have dinner with and why?","What was the last book you read and what do you think about it?",
                             "If you were going on a desert island holiday, what three things would you take with you?","If you could be any movie character, who would you be and why?",
                             "What are the must-have toppings on a pizza for you?","What did you want to be when you were a child and how does it relate to your current profession?"]             
         
    def get_text(self):
        # Get user input from the chat interface
        input_text = st.chat_input("You: ", key="input")
        return input_text
    
    def run(self): 
        st.title("FileWiseGPT🤖")
        instruction = st.text_area(
            "Your prompt",
            value="You are a helpful AI assistant.",
        )
        
        if "history" not in st.session_state:
            st.session_state["history"] = []
        
        if len(st.session_state["history"]) == 0:
            self.display_language_message()
        
        #Changing theme
        """ Doesn't work well yet.
        on = st.toggle('Change Theme')
        if on:
            data = {
                "theme": {
                    "primaryColor": "#060606",
                    "backgroundColor": "#222021",
                    "secondaryBackgroundColor": "#960019",
                    "textColor": "#FFFFFF",
                    "font": "sans serif"
                }
            }
        else:
            data = {
                    "theme": {
                        "primaryColor": "#F63366",
                        "backgroundColor": "#C6E6FB",
                        "secondaryBackgroundColor": "#f9c9c6",
                        "textColor": "#262730",
                        "font": "sans serif"
                    }
                }
        with open('.streamlit/config.toml', 'w') as toml_file:
            toml.dump(data, toml_file)
        """
        
        # Sidebar operations
        self.sidebar_operations()
        for message in st.session_state["history"]:
            if message["role"] == "user":
                with st.chat_message(message["role"],avatar="images/dark_theme_user.png"):
                    st.markdown(message["content"])     
            else:
                with st.chat_message(message["role"],avatar="images/dark_theme_chatbot.png"):
                    st.markdown(message["content"])     
                
        # Get user input
        if self.chatbot.is_uploaded:
            user_input = self.get_text()

            if user_input:
                # Process user input and generate response                
                st.chat_message("user",avatar="images/dark_theme_user.png").markdown(user_input)
                st.session_state["history"].append({"role": "user", "content": user_input})  
                if self.chatbot.get_selected_language() == 'English':
                    random_question = self.wait_arr_eng[random.randint(0,len(self.wait_arr_eng)-1)]
                else:
                    random_question = self.wait_arr_tr[random.randint(0,len(self.wait_arr_tr)-1)]

                with st.spinner(random_question):
                    output = self.chatbot.do_query(user_input)
                    with st.chat_message("assistant",avatar="images/dark_theme_chatbot.png"):
                        message_placeholder = st.empty()
                        full_response = ''
                        for token in output:
                            time.sleep(self.time_arr[random.randint(0,len(self.time_arr)-1)])
                            full_response += token
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                    st.session_state["history"].append({"role": "assistant", "content": full_response})

    def sidebar_operations(self):
        st.sidebar.image("images/last_logo.png", width=250)            
        with st.sidebar:
             # Select the API for the chatbot
            selected_api = st.selectbox('Which embedding model do you want to use?', ('OpenAI', ""))
            self.chatbot.select_api(selected_api)
            
            if selected_api == "OpenAI":
                self.chatbot.set_selected_model("OpenAI")
                
            selected_language = st.selectbox("Choose your language 🌍", ('English', 'Turkish'))
            self.chatbot.set_language(selected_language)            
            
            uploaded_file = st.file_uploader("Choose files 📂", accept_multiple_files=True)
            if uploaded_file is not None:
                self.chatbot.upload_file(uploaded_file)
                
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if openai_api_key:
                self.chatbot.set_openai_api_key(openai_api_key)
                os.environ['OPENAI_API_KEY'] = openai_api_key
                
            temperature = st.slider("Temperature",min_value=0.0,max_value=1.0,step=0.01,value=self.chatbot.temperature)
            self.chatbot.set_temperature(temperature)
            
            presence_penalty = st.slider("Presence Penalty",min_value=-2.0,max_value=2.0,step=0.01,value=self.chatbot.presence_penalty)
            self.chatbot.set_presence_penalty(presence_penalty)
            
            freq_penalty = st.slider("Frequency Penalty",min_value=-2.0,max_value=2.0,step=0.01,value=self.chatbot.frequency_penalty)
            self.chatbot.set_frequency_penalty(freq_penalty)
            
    def display_language_message(self):
        # Determine the message based on the selected language
        if self.chatbot.get_selected_language() == 'English':
            message = "Hi, upload a file before starting the chat. Changing the language only affects my response language, you can ask in any language."
        else:
            message = "Merhaba, sohbete başlamadan önce bir dosya yükleyin. Dili değiştirmek yalnızca benim yanıt dilini etkiler, siz herhangi bir dilde sorabilirsiniz."

        st.session_state["history"].append({"role": "assistant", "content": message})
