import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import streamlit as st

# Function to load and split the PDF
def load_and_split_pdf(uploaded_file):
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())  # Write the uploaded file content
        tmp_file_path = tmp_file.name  # Temporary file path

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # Delete the temporary file after use
    os.remove(tmp_file_path)
    
    return documents

# Function to create a Q&A chain based on the PDF content
def create_qa_chain(documents):
    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings()

    # Store documents in a vector database (ChromaDB)
    db = Chroma.from_documents(documents, embeddings)

    # Create a conversational Q&A chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model="gpt-3.5-turbo"), retriever=db.as_retriever()
    )
    return qa_chain

# Function to handle Q&A interaction
def handle_questions(qa_chain):
    # Maintain chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    question = st.text_input("Ask about the content of the PDF:")

    if question:
        # Add the new question to the history
        st.session_state['chat_history'].append(('User', question))
        
        # Pass the history and question to the Q&A chain
        response = qa_chain.run({
            "question": question,
            "chat_history": st.session_state['chat_history']
        })
        
        # Display the response
        st.write("Answer: ", response)
        
        # Add the response to the history
        st.session_state['chat_history'].append(('Bot', response))

# Streamlit UI setup
st.title("PDF Q&A System")

# Upload PDF
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    # Load and split the content of the PDF
    documents = load_and_split_pdf(pdf_file)
    
    # Create the Q&A system based on the PDF content
    qa_chain = create_qa_chain(documents)
    
    # Interface for Q&A
    st.subheader("Ask questions about the uploaded PDF")
    handle_questions(qa_chain)

else:
    st.info("Please upload a PDF file to get started.")

# UI/UX improvements:
# - Clear and concise language for the user.
# - Helpful instructions like "Please upload a PDF file to get started."
# - Distinct input areas for asking questions and displaying answers.
# - Minimalistic design to avoid unnecessary distractions.
# - Session state is used to retain chat history for continuous interaction.


# Footer for credits or instructions
st.markdown("""
    ---
    Made with 💜 by your friendly AI Assistant.
""")
