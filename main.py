from langchain_community.embeddings import VertexAIEmbeddings
import pinecone
import streamlit as st
from google.oauth2 import service_account
from google.auth.transport.requests import Request

PINECONE_API_KEY = ""
PINECONE_API_ENV = ""
PINECONE_INDEX_NAME = ""
from pinecone import Pinecone
#intialize thre pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Explicitly set credentials using service account key
creds = service_account.Credentials.from_service_account_file(
    '/path/to/your/service-account-key.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Perform a refresh to check if credentials are valid
creds.refresh(Request())

# Now you can use creds for your API calls or any other authentication-required operations
from langchain_google_vertexai.embeddings import VertexAIEmbeddings

# Set the project ID (replace 'your-project-id' with your actual project ID)
project_id = ''

# Initialize VertexAIEmbeddings with the specified project ID
v_embeddings = VertexAIEmbeddings(project=project_id)


def find_match(input_query):
    input_em = v_embeddings.embed_query(input_query)
    result = index.query(vector=input_em, include_metadata=True, top_k=3,include_values=True)  
    result_context = result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']
    return result_context

def create_context_using_documents(docs):
    context_string= ""
    for doc in docs:
        context_string += doc.page_content + "\n\n"
    return context_string

def get_conversation_string():
    conversation_string= ""
    for i in range(len(st.session_state['responses']) -1):
        conversation_string+="Human: "+st.session_state['requests'][i]+"\n"
        conversation_string+="Bot: "+st.session_state['responses'][i+1]+"\n"
    return conversation_string
