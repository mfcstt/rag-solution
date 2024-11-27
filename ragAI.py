#!/usr/bin/env python
# coding: utf-8

# In[10]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain


# In[11]:


import os
openai_api_key = os.getenv("OPENAI_API_KEY")

# In[12]:


embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=200)


# ### Carregar Arquivo PDF

# In[13]:


pdf = "lei_ia_2023.pdf"
loader = PyPDFLoader(pdf, extract_images=False)
pages = loader.load_and_split()


# ### Chunks

# In[14]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=20,
    length_function = len,
    add_start_index = True
)

chuncks = text_splitter.split_documents(pages)


# ### Salvando Chunks no VectorDB

# In[15]:


db = Chroma.from_documents(chuncks, embedding=embeddings_model, persist_directory="chroma_db")
db.persist()


# ### Método Retriver e Chain

# In[16]:


vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings_model)

# Load the retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Construct the QA chain for the chatbot
chain = load_qa_chain(llm, chain_type="stuff")


# ### Execução e Query

# In[17]:


def ask(question):
    context = retriever.get_relevant_documents(question)
    answer = (chain({"input_documents": context, "question": question}, return_only_outputs=True)) ['output_text']
    return answer


# In[18]:


user_question = input("Ask me anything: ")
answer = ask(user_question)
print("Answer:", answer)

