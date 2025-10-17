import tempfile
import streamlit as st
from PIL import Image
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from trail9withoutstreamlit import Trail9  # Your custom PDF extractor class

# ✅ Direct Gemini API key
GOOGLE_API_KEY = "AIzaSyAnjaYecAxdT2nueC9_FzR-_lQoSebNdgY"

# --- Streamlit Configuration ---
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📘 DocuMind")

# --- Step 1: Upload PDF ---
uploaded_pdf = st.file_uploader("📂 Upload your PDF document", type=["pdf"])

if uploaded_pdf:
    st.success("✅ PDF Uploaded Successfully!")

    # Save temporarily for Trail9
    # Creates a temporary file on disk.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read()) # Writes the uploaded PDF content to this temp file.
        pdf_path = tmp_file.name # holds the path to the temporary file

    # --- Step 2: Extract text, images, and tables using Trail9 ---
    with st.spinner("🔍 Extracting content from PDF..."): # Shows a “loading” animation while extraction runs.
        extractor = Trail9(pdf_path) # Initializes your PDF extractor.
        full_text, images, tables, headings = extractor.extract_full_document() # extracts full content in the pdf
        extractor.close() # closes the file

    # --- Step 3: Split text into chunks ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150) # Splits large text into smaller chunks for embeddings.
    chunks = splitter.split_text(full_text)
    st.info(f"📄 Document divided into **{len(chunks)} chunks** for retrieval.") # show how many chunks it is divided into

    # --- Step 4: Build FAISS Vector Store using Hugging Face Embeddings ---
    with st.spinner("⚙️ Creating embeddings and building retriever..."):
        # Converts each chunk into a numeric embedding.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Wraps each chunk as a Document object for FAISS.
        documents = [Document(page_content=chunk) for chunk in chunks]
        vectorstore = FAISS.from_documents(documents, embeddings) # Builds the vector store with embeddings.
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Returns top 3 similar chunks for each query.

    # --- Step 5: Initialize Gemini LLM and QA Chain ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # --- Step 6: Display PDF Contents (Text / Images / Tables) ---
    st.markdown("---")
    st.subheader("📊 PDF Content Overview")

    col1, col2, col3 = st.columns(3) # Creates 3 columns for Text / Images / Tables buttons.

    with col1:
        if st.button("📝 Show Text"): # Shows first 5000 characters of PDF text in a scrollable text area.
            st.text_area("Extracted Text", full_text[:5000], height=300)

    with col2:
        if st.button("🖼️ Show Images"):
            if images: # Shows first 5 images extracted from PDF.
                for idx, (i, image_io, ext) in enumerate(images[:5], start=1):
                    st.image(Image.open(image_io), caption=f"Image {idx} ({ext})", use_column_width=True)
            else:
                st.info("No images found in this document.")

    with col3:
        if st.button("📈 Show Tables"): # Shows first 3 tables in the PDF using st.dataframe.
            if tables:
                for idx, df in enumerate(tables[:3], start=1):
                    st.markdown(f"**Table {idx}:**")
                    st.dataframe(df)
            else:
                st.info("No tables detected in this document.")

    # --- Step 7: Chatbot Section ---
    st.markdown("---")
    st.subheader("💬 Chat with your PDF")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        with st.spinner("🤔 Thinking..."):
            response = qa_chain.run(user_question)

        st.session_state["chat_history"].append(("You", user_question))
        st.session_state["chat_history"].append(("Gemini", response))

    # ✅ Display chat history (latest on top)
    for role, msg in reversed(st.session_state["chat_history"]):
        if role == "You":
            st.markdown(f"**👤 You:** {msg}")
        else:
            st.markdown(f"** Gemini:** {msg}")

else:
    st.info("👆 Upload a PDF to start the chatbot.")
