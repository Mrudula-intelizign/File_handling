import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os

# --- SETUP ---
st.set_page_config(layout="wide")
st.title("📄 Chat with PDF")

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAnjaYecAxdT2nueC9_FzR-_lQoSebNdgY"  # or load from .env

# --- Step 1: Upload PDF ---
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # --- Step 2: Load and chunk text ---
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    st.success(f"✅ Extracted {len(docs)} chunks from your PDF.")

    # --- Step 3: Embedding + Vector Store ---
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    # --- Step 4: Conversational Retrieval Chain ---
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # --- Step 5: Chat Interface ---
    st.subheader("💬 Ask questions about your PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Type your question here:")

    if user_query:
        response = qa_chain.invoke({"question": user_query})
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", response["answer"]))

    # --- Display Chat ---
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑 {sender}:** {msg}")
        else:
            st.markdown(f"**🤖 {sender}:** {msg}")
else:
    st.info("Upload a PDF to begin chatting.")
