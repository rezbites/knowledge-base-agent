import os
import asyncio
import uuid
from typing import Dict, Any, List, Optional

import streamlit as st
from dotenv import load_dotenv
import nest_asyncio

# Import all necessary functions from brain.py
from brain import get_index_for_pdf

from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# 0. Configuration and Initialization
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Set the page configuration for a wider, cleaner look
st.set_page_config(
    page_title="Knowledge Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is not set in the environment. "
        "Please add it to your .env file as GOOGLE_API_KEY=your_gemini_key_here."
    )

# Ensure event loop exists for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()

# -----------------------------
# Session State Initialization and Chat Management Functions
# -----------------------------
if "chats" not in st.session_state:
    default_chat_id = str(uuid.uuid4())
    st.session_state["chats"] = {
        default_chat_id: {
            "title": "New Chat 1",
            "history": [{"role": "system", "content": "none"}],
            "vectordb": None,
        }
    }
    st.session_state["current_chat_id"] = default_chat_id
if "show_ingestion_modal" not in st.session_state:
    st.session_state["show_ingestion_modal"] = False
# New state to track which chat is currently being edited
if "editing_chat_id" not in st.session_state:
    st.session_state["editing_chat_id"] = None

def new_chat():
    """Create a new chat entry and set it as current."""
    new_id = str(uuid.uuid4())
    count = len(st.session_state["chats"]) + 1
    st.session_state["chats"][new_id] = {
        "title": f"New Chat {count}",
        "history": [{"role": "system", "content": "none"}],
        "vectordb": None,
    }
    st.session_state["current_chat_id"] = new_id

@st.cache_resource
def create_vectordb(files: List[st.runtime.uploaded_file_manager.UploadedFile], filenames: List[str]):
    """
    Build a FAISS vector DB from PDF files using Google embeddings.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing; cannot build vector DB.")
    with st.spinner("Building vector database from PDFs..."):
        return get_index_for_pdf(
            [file.getvalue() for file in files], filenames, GOOGLE_API_KEY
        )

@st.cache_data
def get_relevant_context(_vectordb, question: str, k: int = 2):
    """
    Cached function to retrieve relevant context from vector DB.
    Reduced k from 3 to 2 for faster retrieval.
    """
    search_results = _vectordb.similarity_search(question, k=k)
    return "\n\n".join([result.page_content for result in search_results])

# Get the mutable reference to the current chat dictionary
current_chat: Dict[str, Any] = st.session_state["chats"][st.session_state["current_chat_id"]]


# -----------------------------
# 1. Sidebar: Chat History Only (with Edit Functionality)
# -----------------------------
with st.sidebar:
    st.markdown("## 🧠 Knowledge Agent") # Main Heading for the Agent
    
    # New Chat Button
    if st.button("➕ Create New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.subheader("Chat History")
    # Display historical chats and allow switching (newest first)
    chat_ids = list(st.session_state["chats"].keys())
    
    # Custom function to handle renaming logic
    def rename_chat(chat_id, new_title):
        if new_title.strip():
            st.session_state["chats"][chat_id]["title"] = new_title.strip()
        st.session_state["editing_chat_id"] = None
        st.rerun()

    for chat_id in reversed(chat_ids):
        title = st.session_state["chats"][chat_id]["title"]
        is_current = st.session_state["current_chat_id"] == chat_id
        is_editing = st.session_state["editing_chat_id"] == chat_id

        if is_editing:
            # Show text input for editing
            with st.container():
                new_title = st.text_input(
                    "Edit Name",
                    value=title,
                    key=f"edit_input_{chat_id}",
                    label_visibility="collapsed"
                )
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save", key=f"save_{chat_id}", use_container_width=True):
                        rename_chat(chat_id, new_title)
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_{chat_id}", use_container_width=True):
                        st.session_state["editing_chat_id"] = None
                        st.rerun()
        else:
            # Default display mode
            col_chat, col_edit = st.columns([0.8, 0.2])
            
            with col_chat:
                if is_current:
                    # Current chat is displayed as highlighted text/button
                    st.button(f"**{title}**", key=f"hist_{chat_id}", use_container_width=True)
                else:
                    # Non-current chat is a clickable button to switch
                    if st.button(title, key=f"hist_{chat_id}", use_container_width=True):
                        st.session_state["current_chat_id"] = chat_id
                        st.rerun()

            with col_edit:
                # Button to initiate editing mode
                if st.button("✏️", key=f"edit_{chat_id}", help="Rename chat"):
                    st.session_state["editing_chat_id"] = chat_id
                    st.rerun()
    
    st.markdown("---")


# -----------------------------
# 2. Main Chat Interface and Ingestion Modal (Popup)
# -----------------------------
current_history = current_chat["history"]
current_vectordb = current_chat["vectordb"]

# Custom CSS for centering and a clean look
st.markdown(
    """
    <style>
    .stApp { padding-top: 2rem !important; }
    .centered-text {
        text-align: center;
        font-size: 3em;
        font-weight: 300;
        margin-bottom: 2em;
        color: #ddd;
    }
    .stButton>button {
        width: 100%;
        margin: 0 auto;
        display: block;
    }
    .center-button-container {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 10px;
    }
    .modal-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin-top: -100px;
    }
    .modal-hr {
        margin: 10px 0 !important;
        opacity: 0.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to run ingestion logic (called by the button)
def run_ingestion(pdf_files):
    errors = []
    success_msgs = []
    new_vectordb = current_chat["vectordb"]
    
    # 1. Handle PDF ingestion
    if pdf_files:
        try:
            pdf_file_names = [file.name for file in pdf_files]
            new_vectordb = create_vectordb(pdf_files, pdf_file_names)
            success_msgs.append(f"Indexed {len(pdf_file_names)} PDF(s).")
        except Exception as e:
            errors.append(f"PDF ingestion failed: {e}")
    
    # Update chat state and hide modal ONLY IF successful
    if new_vectordb is not None and pdf_files:
        current_chat["vectordb"] = new_vectordb
        st.session_state["show_ingestion_modal"] = False
        # Immediately rerun to show chat interface
        st.rerun()
    elif not pdf_files:
        st.warning("No PDF files provided to ingest.")
    
    # Show errors if ingestion failed but sources were provided
    if errors:
        for err in errors: st.error(err)
    
# Display modal - more specific condition
if st.session_state.get("show_ingestion_modal", False) or (current_vectordb is None and len(current_history) <= 1):
    col_empty1, col_modal, col_empty2 = st.columns([1, 2, 1])
    
    with col_modal:
        with st.container(border=True):
            st.header("🧠 Knowledge Agent: Empower Your Chat")
            st.write("Upload PDF documents to build the knowledge base for the current conversation.")
            
            st.markdown('<hr class="modal-hr">', unsafe_allow_html=True)
            
            pdf_files = st.file_uploader(
                "Upload PDF documents",
                type="pdf",
                accept_multiple_files=True,
                key="modal_pdf_upload",
            )
            
            st.markdown('<hr class="modal-hr">', unsafe_allow_html=True)
            
            col_left, col_center, col_right = st.columns([1, 2, 1])

            with col_center:
                if st.button("Submit", use_container_width=True, key="ingest_button_modal"):
                    run_ingestion(pdf_files)
            
            if current_chat["vectordb"] is not None:
                st.markdown("")
                col_cancel_left, col_cancel_center, col_cancel_right = st.columns([1, 2, 1])
                with col_cancel_center:
                    if st.button("Cancel", use_container_width=True, key="cancel_button_modal_2"):
                        st.session_state["show_ingestion_modal"] = False
                        st.rerun()
            else:
                st.markdown("")

    st.stop()


# Show chat interface
if len(current_history) <= 1:
    st.markdown("<div class='centered-text'>Start your knowledge journey here.</div>", unsafe_allow_html=True)
else:
    # Show past chat history
    for message in current_history[1:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# -----------------------------
# 3. Chat Input and Logic
# -----------------------------

# Use st.form to capture the input
with st.form(key='chat_form', clear_on_submit=True):
    col_plus, col_input, col_send = st.columns([0.1, 0.8, 0.1])
    
    with col_plus:
        if st.form_submit_button("➕", help="Add new sources/documents"):
            st.session_state["show_ingestion_modal"] = True
            st.rerun()
            
    with col_input:
        question_text = st.text_input("Ask anything", key="question_text_input", label_visibility="collapsed")
        
    with col_send:
        submitted = st.form_submit_button("➤")


# Add a flag to prevent rerun loops
if "processing_query" not in st.session_state:
    st.session_state["processing_query"] = False

# Process the question
if submitted and question_text and not st.session_state["processing_query"]:
    question = question_text
    st.session_state["processing_query"] = True
    
    if not current_vectordb:
        with st.chat_message("assistant"):
            st.write("Please ingest sources using the **➕** button or the central prompt.")
        st.session_state["processing_query"] = False
    else:
        # Retrieve context with caching
        try:
            pdf_extract = get_relevant_context(current_vectordb, question, k=2)
        except Exception as e:
            with st.chat_message("assistant"):
                st.write(f"Error retrieving context from vector DB: {e}")
        else:
            # Simplified, shorter prompt for faster processing
            prompt_template = """You are a Knowledge Base Assistant. Answer using ONLY the provided documents.

Rules:
1. Use document information and cite: (filename: <name>, page: <number>)
2. If not in documents, state clearly and give brief general info if appropriate
3. Be concise and accurate

Documents:
{pdf_extract}

Answer the question using only this information."""

            system_prompt = prompt_template.format(pdf_extract=pdf_extract)
            current_history[0] = {"role": "system", "content": system_prompt}

            # Add user message
            current_history.append({"role": "user", "content": question})
            
            # Echo user message
            with st.chat_message("user"):
                st.write(question)

            # Call Gemini with streaming for instant feedback
            with st.chat_message("assistant"):
                botmsg = st.empty()
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-exp",
                        google_api_key=GOOGLE_API_KEY,
                        temperature=0.2,
                        max_tokens=512,  # Limit response length for speed
                        streaming=True,
                    )
                    
                    # Stream the response for instant feedback
                    answer_text = ""
                    for chunk in llm.stream(current_history):
                        if hasattr(chunk, "content"):
                            answer_text += chunk.content
                            botmsg.write(answer_text)
                        
                except Exception as e:
                    answer_text = f"Error calling Gemini model: {e}"
                    botmsg.write(answer_text)

            # Save assistant reply
            current_history.append({"role": "assistant", "content": answer_text})
            
            # Update chat title if first question
            if len(current_history) == 3 and current_chat["title"].startswith("New Chat"):
                current_chat["title"] = question[:25] + "..."
            
            # Reset processing flag and rerun to show the updated chat
            st.session_state["processing_query"] = False
            st.rerun()