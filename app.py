# Q&A System with DeepSeek API
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import hashlib
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Environment Setup
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# UI Configuration
with st.sidebar:
    st.header("📄 Document Settings")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files (multiple supported)",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000)
    overlap_size = st.slider("Overlap Size (characters)", 0, 300, 100)
    
    temperature = st.slider(
        "Response Temperature (0=Strict, 1=Creative)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )

    history_limit = st.slider("Chat History Limit", 3, 30, 5)
    
    # Model Selection 
    st.header("🚀 Model Settings")
    model_choice = st.selectbox(
        "Select AI Model",
        options=["DeepSeek-V3", "DeepSeek-R1"],
        index=0,
        help="V3: General purpose, R1: Enhanced reasoning"
    )
    model_id_mapping = {
        "DeepSeek-V3": ("Pro/deepseek-ai/DeepSeek-V3", "V3"),
        "DeepSeek-R1": ("Pro/deepseek-ai/DeepSeek-R1", "R1")
    }
    st.session_state.selected_model, model_version = model_id_mapping[model_choice]

# Document Processing
def process_documents(files):
    """Extract text and tables with structure preservation"""
    docs = {}
    
    header_pattern = re.compile(
        r'(\n\s*[A-Z][\w\s]+\b(?:\.\s*\n|\s*\n{2,}))|'  # Title patterns
        r'(\n\d+\.\d+\s+.+?\n)|'  # Numbered sections
        r'(\n#+\s+.+?\n)|'  # Markdown headers
        r'(\n•\s+.+?\n)'   # Bullet points
    )

    for file in files:
        all_text = []
        
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            page_texts = [page.get_text("text") for page in doc]
        
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                    table_texts = [
                        f"\n[TABLE_START]\n{' | '.join(map(str, row))}\n[TABLE_END]\n"
                        for table in tables for row in table
                    ]
                    
                    full_page_text = page_texts[page_num] 
                    if table_texts:
                        full_page_text += "\n\n### TABLE CONTENT ###\n" + "\n".join(table_texts)
                        
                    all_text.append(full_page_text)
                except Exception as e:
                    st.warning(f"Page {page_num+1} error: {str(e)}")
                    all_text.append(page_texts[page_num])

        combined_text = "\n\n".join(all_text)
        combined_text = header_pattern.sub(r'\n###### \g<0>', combined_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=[
                "\n###### ",
                "\n\n## ",
                "\n\n# ",
                "\nTABLE_END",
                "\n\n",
                "\n",
                ". ",
                " ",
            ],
            keep_separator=True
        )
        
        chunks = text_splitter.split_text(combined_text)
        chunks = [chunk.replace("###### ", "")
                     .replace("[TABLE_START]", "\n[TABLE]:\n")
                     .replace("[TABLE_END]", "\n[END TABLE]")
                  for chunk in chunks]
        
        docs[file.name] = chunks
    
    return docs

# Context Retrieval
def find_relevant_chunks(query, docs, top_k=3):
    """TF-IDF based text chunk retrieval"""
    all_chunks = [chunk for doc_chunks in docs.values() for chunk in doc_chunks]
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform(all_chunks + [query])
    query_vector = tfidf_matrix[-1]
    
    similarities = np.dot(tfidf_matrix[:-1], query_vector.T).toarray().flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [all_chunks[i] for i in top_indices]

# Main Interface
st.title("📚 DeepSeek Document Q&A System")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": f"You are DeepSeek {model_version} Document Analyst"},
        {"role": "assistant", "content": "Document analysis ready - Upload PDFs and ask questions"}
    ]

# Display message history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Document Handling
if uploaded_files:
    current_hash = hashlib.md5(str(uploaded_files).encode()).hexdigest()
    if "file_hash" not in st.session_state or st.session_state.file_hash != current_hash:
        with st.spinner("🔍 Processing documents..."):
            st.session_state.docs = process_documents(uploaded_files)
            st.session_state.file_hash = current_hash
        st.success(f"✅ Processed {len(uploaded_files)} files, created {sum(len(v) for v in st.session_state.docs.values())} context chunks")

    # Document scope selector
    with st.sidebar:
        selected_doc = st.selectbox("Search Scope", ["All Documents"] + list(st.session_state.docs.keys()))

# Query Processing
if user_query := st.chat_input("Ask your question"):
    if not uploaded_files:
        st.warning("⚠️ Please upload PDF documents first")
        st.stop()
    
    with st.chat_message("user"):
        st.markdown(f"**User:** {user_query}")
    
    st.session_state.messages.append({"role": "user", "content": user_query})

    search_docs = st.session_state.docs if selected_doc == "All Documents" else {
        selected_doc: st.session_state.docs[selected_doc]
    }

    relevant_chunks = find_relevant_chunks(user_query, search_docs)

    system_prompt = {
        "role": "system",
        "content": f'''You are DeepSeek AI Assistant ({model_choice} version). Key instructions:
1. When asked about your identity: "I'm an AI assistant developed by DeepSeek, using our proprietary {model_choice} model."
2. Always reference: 
{''.join(relevant_chunks)}

Response requirements:
1. Prioritize table data
2. Maintain numerical precision
3. Clearly indicate when information is missing
4. Use structured formats when appropriate'''
    }

    try:
        with st.chat_message("assistant"):
            response_stream = client.chat.completions.create(
                model=st.session_state.selected_model,
                messages=[system_prompt] + st.session_state.messages[-history_limit:],
                temperature=temperature,
                stream=True
            )
            
            full_response = ""
            response_container = st.empty()
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_container.markdown(f"**AI Assistant:** {full_response}▌")
            response_container.markdown(f"**AI Assistant:** {full_response}")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    except Exception as error:
        st.error(f"🚨 API Error: {str(error)}")
        st.session_state.messages.pop()
