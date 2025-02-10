import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Updated template for summarization
template = """
You are a professional document summarizer. Analyze the following document content and generate a concise summary that captures the key points, main ideas, and critical information. Structure your summary in clear paragraphs and avoid technical jargon. Keep it under 150 words.

Document content: {context}

Summary:
"""

pdfs_directory = './pdf/'

embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:1.5b")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def generate_summary(context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"context": context})

# Streamlit UI
uploaded_file = st.file_uploader(
    "Upload PDF for summarization",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    # Process PDF
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)
    
    # Generate summary from all chunks
    full_context = "\n\n".join([doc.page_content for doc in chunked_documents])
    with st.spinner("Analyzing document and generating summary..."):
        summary = generate_summary(full_context)
    
    # Display results
    st.subheader("Document Summary")
    st.write(summary)
    
    # Optional: Show processing details
    with st.expander("Show document chunks"):
        st.write(f"Total chunks: {len(chunked_documents)}")
        for i, chunk in enumerate(chunked_documents):
            st.write(f"Chunk {i+1}: {chunk.page_content[:100]}...")