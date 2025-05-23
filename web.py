from langchain_community.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_google_vertexai import ChatVertexAI
import streamlit as st
from streamlit_chat import message
from main import * 

st.subheader("PDF-Interacter")

# Initialize responses and requests if not present in session state
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I help you ?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize buffer_memory if not present in session state
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

llm = ChatVertexAI()

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="Answer the question as truthfully as possible, using the provided context, and if the answer is not "
             "contained within the text below, say 'I don't know'")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages(
    [
        system_msg_template,
        MessagesPlaceholder(variable_name="history"),
        human_msg_template
    ]
)

# Create ConversationChain with initialized buffer_memory
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            context = find_match(query)
            prompt_prepare = """Client Query:
            {query}
            
            Context:
            {context}
            """.format(query=query, context=context)
            response = conversation.predict(input=prompt_prepare)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
