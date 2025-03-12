# -*- coding: utf-8 -*-
# Document Q&A System with DeepSeek API
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

# ================== Load Environment Variables ==================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# ================== Streamlit Sidebar Configuration ==================
with st.sidebar:
    st.header("📄 Document Configuration")
    
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

    history_limit = st.slider("Chat History Limit", 3, 20, 5)

# ================== Enhanced Document Processing ==================
def process_documents(files):
    """Process PDFs with text and table extraction, preserving document structure"""
    docs = {}
    
    # Regular expression patterns for section headers
    header_pattern = re.compile(
        r'(\n\s*[A-Z][\w\s]+\b(?:\.\s*\n|\s*\n{2,}))|'  # Capitalized titles
        r'(\n\d+\.\d+\s+.+?\n)|'  # Numbered sections
        r'(\n#+\s+.+?\n)|'  # Markdown-style headers
        r'(\n•\s+.+?\n)'   # Bullet point headers
    )

    for file in files:
        all_text = []
        
        # Extract text structure using PyMuPDF
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            page_texts = []
            for page in doc:
                page_text = page.get_text("text")
                page_texts.append(page_text)
        
        # Refine table extraction using pdfplumber
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract tables with enhanced detection
                    tables = page.extract_tables()
                    table_texts = []
                    for table in tables:
                        # Preserve table structure with vertical alignment
                        text = "\n".join([" | ".join(map(str, row)) for row in table])
                        table_texts.append(f"\n[TABLE_START]\n{text}\n[TABLE_END]\n")
                    
                    # Merge text and tables
                    full_page_text = page_texts[page_num] 
                    if table_texts:
                        full_page_text += "\n\n### Table Content ###\n" + "\n".join(table_texts)
                        
                    all_text.append(full_page_text)
                except Exception as e:
                    st.warning(f"Page {page_num+1} processing failed: {str(e)}")
                    all_text.append(page_texts[page_num])

        # Post-processing
        combined_text = "\n\n".join(all_text)
        combined_text = header_pattern.sub(r'\n###### \g<0>', combined_text)

        # Hierarchical text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=[
                "\n###### ",  # Custom header markers
                "\n\n## ",    # Section headers
                "\n\n# ",     # Main headers
                "\nTABLE_END",# Table boundaries
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                ". ",         # Sentence endings
                " ",          # Word boundaries
            ],
            keep_separator=True
        )
        
        chunks = text_splitter.split_text(combined_text)
        # Cleanup markers and enhance readability
        chunks = [chunk.replace("###### ", "")
                     .replace("[TABLE_START]", "\nTable Content:\n")
                     .replace("[TABLE_END]", "\nTable End")
                  for chunk in chunks]
        
        docs[file.name] = chunks
    
    return docs

# ================== Contextual Chunk Retrieval ==================
def find_relevant_chunks(query, docs, top_k=3):
    """Retrieve relevant text chunks using TF-IDF vectorization"""
    all_chunks = [chunk for doc_chunks in docs.values() for chunk in doc_chunks]
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform(all_chunks + [query])
    query_vector = tfidf_matrix[-1]
    
    similarities = np.dot(tfidf_matrix[:-1], query_vector.T).toarray().flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [all_chunks[i] for i in top_indices]

# ================== Main Interface ==================
st.title("📚 DeepSeek Document Q&A Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ready to analyze documents - Please upload PDF files and ask your question"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================== Document Handling ==================
if uploaded_files:
    current_hash = hashlib.md5(str(uploaded_files).encode()).hexdigest()
    if "file_hash" not in st.session_state or st.session_state.file_hash != current_hash:
        with st.spinner("🔍 Processing documents..."):
            st.session_state.docs = process_documents(uploaded_files)
            st.session_state.file_hash = current_hash
        st.success(f"✅ Successfully processed {len(uploaded_files)} files, created {sum(len(v) for v in st.session_state.docs.values())} context-aware chunks")

    # Document selector
    with st.sidebar:
        selected_doc = st.selectbox("Search Scope", ["All Documents"] + list(st.session_state.docs.keys()))

# ================== Query Handling ==================
if user_query := st.chat_input("Enter your question"):
    if not uploaded_files:
        st.warning("⚠️ Please upload PDF documents first")
        st.stop()
    
    with st.chat_message("user"):
        st.markdown(f"**User:** {user_query}")
    
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Determine search scope
    search_docs = st.session_state.docs if selected_doc == "All Documents" else {
        selected_doc: st.session_state.docs[selected_doc]
    }

    relevant_chunks = find_relevant_chunks(user_query, search_docs)

    # System prompt with table handling instructions
    system_prompt = {
        "role": "system",
        "content": f'''Analyze the following document sections (including tables):
{''.join(relevant_chunks)}

Response requirements:
1. Prioritize data from tables when applicable
2. Maintain numerical precision (no rounding)
3. Clearly state if information is insufficient'''
    }

    try:
        with st.chat_message("assistant"):
            response_stream = client.chat.completions.create(
                model="Pro/deepseek-ai/DeepSeek-R1",
                messages=[system_prompt] + st.session_state.messages[-history_limit:],
                temperature=temperature,
                stream=True
            )
            
            full_response = ""
            response_container = st.empty()
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_container.markdown(f"**AI:** {full_response}▌")
            response_container.markdown(f"**AI:** {full_response}")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    except Exception as error:
        st.error(f"🚨 Request failed: {str(error)}")
        st.session_state.messages.pop()
