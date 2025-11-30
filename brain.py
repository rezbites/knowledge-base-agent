import re
from io import BytesIO
from typing import Tuple, List, Optional

# FIXED: Updated import for newer LangChain versions
from langchain_core.documents import Document 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from pypdf import PdfReader

# -----------------------------
# PDF processing helpers
# -----------------------------

def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    """
    Parse a PDF file into a list of page texts.
    Returns (list_of_page_texts, filename).

    Basic cleaning is applied: de-hyphenation, line joining, etc.
    """
    texts: List[str] = []
    try:
        pdf = PdfReader(file)
    except Exception as e:
        raise ValueError(f"Failed to read PDF '{filename}': {e}")

    for page_idx, page in enumerate(pdf.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            # Skip this page but continue others
            print(f"[parse_pdf] Warning: failed to extract page {page_idx+1} of '{filename}': {e}")
            continue

        if text is None:
            continue

        # Basic cleanup
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        texts.append(text)

    if not texts:
        raise ValueError(f"No text could be extracted from PDF '{filename}'")

    return texts, filename


def text_to_docs(text: List[str], filename: str, source_url: Optional[str] = None) -> List[Document]:
    """
    Convert a list of page texts into LangChain Document chunks with metadata:
    - page (1-based)
    - chunk index
    - filename
    - optional url (if source_url is provided)
    """
    if isinstance(text, str):
        text = [text]

    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks: List[Document] = []

    for page_doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(page_doc.page_content)

        for i, chunk in enumerate(chunks):
            metadata = {
                "page": page_doc.metadata["page"],
                "chunk": i,
                "filename": filename,
            }
            if source_url:
                metadata["url"] = source_url

            chunk_doc = Document(page_content=chunk, metadata=metadata)
            # Convenient combined source id
            chunk_doc.metadata["source"] = f"{metadata['page']}-{metadata['chunk']}"
            doc_chunks.append(chunk_doc)

    if not doc_chunks:
        raise ValueError(f"No document chunks created for '{filename}'")

    return doc_chunks


def docs_to_index(docs: List[Document], google_api_key: str) -> FAISS:
    """
    Create a FAISS vector index from documents using Google Generative AI embeddings.
    """
    if not docs:
        raise ValueError("No documents provided to build index")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            # Using the modern embedding model
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}")

    try:
        index = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        # Re-raise with a more specific error related to ingestion failure
        raise ValueError(f"Failed to create FAISS index (check API key/embedding quota): {e}")

    return index


def get_index_for_pdf(pdf_files: List[bytes], pdf_names: List[str], google_api_key: str) -> FAISS:
    """
    Build a FAISS index for multiple PDFs.
    pdf_files: list of raw bytes for each PDF
    pdf_names: corresponding list of filenames
    """
    if not pdf_files or not pdf_names:
        raise ValueError("No PDF files or filenames provided")
    if len(pdf_files) != len(pdf_names):
        raise ValueError("pdf_files and pdf_names must have the same length")

    documents: List[Document] = []

    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        try:
            text_pages, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
            doc_chunks = text_to_docs(text_pages, filename)
            documents.extend(doc_chunks)
        except Exception as e:
            print(f"[get_index_for_pdf] Warning: skipping '{pdf_name}' due to error: {e}")

    if not documents:
        raise ValueError("No documents extracted from any PDFs")

    index = docs_to_index(documents, google_api_key)
    return index