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
        self.wait_arr_tr = ["Bunu biliyor muydunuz? Penguenlerin bir kralı vardır! Norveç 2008 yılında bir pengueni \"Sir Nils Olav\" adıyla şövalye ilan etti.",
                            "Bunu biliyor muydunuz? Domates aslında bir meyvedir, ancak 1893'te ABD Yüksek Mahkemesi onu sebze olarak sınıflandırdı!",
                            "Bunu biliyor muydunuz? Dünyanın en kısa savaşı 1896'da İngiltere ve Zanzibar arasında oldu ve sadece 38 dakika sürdü.",
                            "Bunu biliyor muydunuz? Ay'ın üzerinde mülk satın alabilirsiniz! Evet, gerçek olmasa da, bazı şirketler Ay'da \"arazi\" satıyor.",
                            "Bunu biliyor muydunuz? Dünyanın en uzun yer ismi Galler'de ve tam 58 harften oluşuyor: Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch.",
                            "Bunu biliyor muydunuz? Bir zamanlar Avustralya'da \"Emu Savaşı\" vardı. 1932'de Avustralya hükümeti, emu kuşlarına karşı bir askeri operasyon başlattı!",
                            "Bunu biliyor muydunuz? Balıklar susuz kalabilir. Evet, yanlış duymadınız. Tatlı su balıkları, suda yeterince tuz bulamazlarsa \"susuzluktan\" ölebilirler.",
                            "Bunu biliyor muydunuz? Venedik'te gondolların boyutu standarttır çünkü tüm kanallardan geçebilmeleri için belirli bir genişliğe sahip olmaları gerekiyor.",
                            "Bunu biliyor muydunuz? Her yıl yaklaşık 1.000 mektup, İtalya'da \"Juliet'in Evi\"ne gönderiliyor. Bu mektuplar, Shakespeare'in ünlü eseri \"Romeo ve Juliet\"e atıfta bulunuyor.",
                            "Bunu biliyor muydunuz? Dünya üzerindeki en eski \"güvenlik kamera\" kaydı 1941'den kalma ve Hitler'in Paris'i ziyaretini gösteriyor."
                            ]
        self.wait_arr_eng = [
                            "Did you know? Penguins have a king! In 2008, Norway knighted a penguin named 'Sir Nils Olav'.",
                            "Did you know? Tomatoes are actually a fruit, but in 1893, the U.S. Supreme Court classified them as a vegetable!",
                            "Did you know? The shortest war in history was between England and Zanzibar in 1896, lasting only 38 minutes.",
                            "Did you know? You can buy property on the Moon! Though not legally recognized, some companies sell 'land' on the Moon.",
                            "Did you know? The world's longest place name is in Wales and has 58 letters: Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch.",
                            "Did you know? There was once an 'Emu War' in Australia. In 1932, the Australian government launched a military operation against emus!",
                            "Did you know? Fish can get thirsty. Yes, you heard right. Freshwater fish can 'die of thirst' if they don't get enough salt in the water.",
                            "Did you know? Gondolas in Venice have a standard size because they need to be a certain width to navigate all the canals.",
                            "Did you know? Each year, about 1,000 letters are sent to 'Juliet's House' in Italy, referencing Shakespeare's famous play 'Romeo and Juliet.'",
                            "Did you know? The world's oldest 'security camera' footage is from 1941 and shows Hitler's visit to Paris."
                            ]
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
        self.chatbot.set_prompt_instruction(instruction)
        
        #Changing theme
        #Doesn't work well yet.
        """ 
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
        if "history" not in st.session_state:
            st.session_state["history"] = []
        
        if len(st.session_state["history"]) == 0:
            self.display_language_message()
            
        st.sidebar.image("images/last_logo.png", width=250)            

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
                    output = self.chatbot._query(user_input)
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
        with st.sidebar:            
            if not self.chatbot.get_selected_language() == "Turkish":
                selected_language = st.selectbox("Choose your language 🌍", ('English', 'Turkish'))
                self.chatbot.set_language(selected_language)     
            else:
                selected_language = st.selectbox("Dilinizi seçiniz 🌍", ('Ingilizce', 'Türkçe'))
                self.chatbot.set_language(selected_language)   
            
             # Select the API for the chatbot
            if not self.chatbot.get_selected_language() == "Turkish":
                selected_api = st.selectbox('Which embedding model do you want to use?', ('OpenAI', ""))
                self.chatbot.select_api(selected_api)
            else:
                selected_api = st.selectbox('Hangi modeli kullanmak istersiniz?', ('OpenAI', ""))
                self.chatbot.select_api(selected_api)
                
            if selected_api == "OpenAI":
                self.chatbot.set_selected_model("OpenAI")  
            
            if not self.chatbot.get_selected_language() == "Turkish":
                uploaded_file = st.file_uploader("Choose files 📂", accept_multiple_files=True)
                if uploaded_file is not None:
                    self.chatbot.upload_file(uploaded_file)
            else:
                uploaded_file = st.file_uploader("Dosya seçiniz 📂", accept_multiple_files=True)
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
            
            top_p = st.slider("Top_P",min_value=-0.0,max_value=1.0,step=0.01,value=self.chatbot.top_p)
            self.chatbot.set_top_p(top_p)
            
            if self.chatbot.get_selected_language() == "Turkish":
                with st.expander("Açıklamaları Gör"):
                    st.write("Temperature: Dil modelinin ne kadar \"yaratıcı\" olacağını belirler. Düşük değer, daha tahmin edilebilir yanıtlar verir. Yüksek değer, daha sıra dışı ve yaratıcı yanıtlara yol açar.")
                    st.write("Presence Penalty: Modelin daha önce kullandığı kelimeleri tekrar kullanmasını önler. Yüksek bir değer, modelin daha önce kullanılmamış kelimeleri kullanmasını teşvik eder.")
                    st.write("Frequency Penalty: Modelin aynı kelimeyi tekrar tekrar kullanmasını önler. Bu değer yükseldikçe, modelin aynı kelimeyi kullanma olasılığı azalır.")
                    st.write("Top P: Modelin yanıt oluştururken kaç farklı olasılığı göz önünde bulunduracağını belirler. Daha yüksek değerler, modelin daha fazla kelime seçeneğinden birini seçmesine izin verir, bu da daha çeşitli yanıtlara yol açar.")
            else:
                with st.expander("See explanation"):
                    st.write("Temperature: Controls how \"creative\" the language model is. A lower value leads to more predictable responses. A higher value makes responses more unusual and creative.")
                    st.write("Presence Penalty: Discourages the model from reusing words it has already used. A higher value encourages the model to use new words that haven't been used before.")
                    st.write("Frequency Penalty: Prevents the model from repeating the same words too often. As this value increases, the likelihood of the model repeating the same word decreases.")
                    st.write("Top P: Determines how many different possibilities the model considers when generating a response. Higher values allow the model to choose from a wider range of word options, leading to more varied responses.")
                    
    def display_language_message(self):
        # Determine the message based on the selected language
        if self.chatbot.get_selected_language() == 'English':
            message = "Hi, upload a file before starting the chat. Changing the language only affects my response language, you can ask in any language."
        else:
            message = "Merhaba, sohbete başlamadan önce bir dosya yükleyin. Dili değiştirmek yalnızca benim yanıt dilini etkiler, siz herhangi bir dilde sorabilirsiniz."

        st.session_state["history"].append({"role": "assistant", "content": message})
