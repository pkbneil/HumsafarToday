import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Load environment variables from secrets management
openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]
langchain_api_key = st.secrets["langchain_api"]["LANGCHAIN_API_KEY"]
langchain_project_name = st.secrets["langchain_project"]["LANGCHAIN_PROJECT"]

os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = langchain_project_name

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('AccentureProjectInformation.pdf')
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=10)
documents = text_splitter.split_documents(docs)

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS

vectorStoreDb = FAISS.from_documents(documents, embeddings)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o')

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Define a retriever from the vector db
retriever = vectorStoreDb.as_retriever()

from langchain_core.output_parsers import StrOutputParser

custom_prompt = PromptTemplate(
    template="""You are an expert assistant. Below is some context from Accenture's documents:

    {context}

    Based on this information, please answer the following question:

    Question: {question}
    Answer: """,
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

## Streamlit code

# Add a logo at the top
st.image('FinalLogo_Accenture.jpg', width=100)  # Replace with the path to your logo

st.title('Humsafar: Your Hyper-Personalized AI Chatbot')

st.markdown("""
Welcome to the Accenture Humsafar Chatbot! This AI-powered assistant is here to help you with various aspects of well-being and professional growth.
""")

# Dropdown menu for the six aspects
option = st.selectbox(
    'Select a category for your query:',
    ('Physical', 'Emotional', 'Relational', 'Financial', 'Purposeful', 'Employable', 'Miscellaneous')
)

# Display the selected aspect
st.write(f'You selected: {option}')

# Input field for user query
input_text = st.text_input('Enter your prompt!')

# Conditional execution based on selected aspect and input
if input_text:
    # Here, you could customize the query handling based on the selected option if needed.
    response = rag_chain.invoke(f'{input_text}')
    st.write(response)

st.markdown("""
*Disclaimer: This chatbot is an internal tool designed for informational purposes. For personalized advice, please consult a professional.*
""")


