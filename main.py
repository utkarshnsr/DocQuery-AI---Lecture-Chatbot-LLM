import streamlit as st
import responseGenerator 
from responseGenerator import ResponseGeneratorClass

def main():
    st.markdown("<h1 style='text-align: center;'>Lecture Notes Q&A Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>This application helps you upload your lecture file (in PDF) and ask the chatbot questions pertaining to the lecture.</h6>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])


    with col1:
        st.header("File Uploader")
        pdf = st.file_uploader("Please upload your lecture note!")
        if pdf:
            responseGen = ResponseGeneratorClass(pdf)
            responseGen.initiateRAGProcess()
            st.success("Successfully uploaded your notes!")
            openChatbot(col2,responseGen)
        
    
def openChatbot(col2, responseGenInstance):
    with col2:
        st.header("Chatbot")
        st.markdown(
            """
            <style>
            .chatbox {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 10px;
            }
            .user, .assistant {
                display: flex;
                margin: 10px 0;
            }
            .user .message, .assistant .message {
                padding: 10px;
                border-radius: 10px;
                max-width: 80%;
            }
            .user .message {
                background-color: #e1ffc7;
                align-self: flex-end;
            }
            .assistant .message {
                background-color: #dcdcdc;
                align-self: flex-start;
            }
            .user {
                justify-content: flex-end;
            }
            .assistant {
                justify-content: flex-start;
            }
            </style>
            """, unsafe_allow_html=True
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        prompt = st.chat_input("Ask your question!")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = responseGenInstance.userInputProcess(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})

        chat_history = "<div id='chatbox' class='chatbox'>"
        for message in st.session_state.messages:
            if message['role'] == 'user':
                chat_history += f"<div class='user'><div class='message'>{message['content']}</div></div>"
            else:
                chat_history += f"<div class='assistant'><div class='message'>{message['content']}</div></div>"
        chat_history += "</div>"


        st.markdown(chat_history, unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
