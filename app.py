# Import necessary libraries
import streamlit as st
from brain import get_index_for_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio

# Load environment variables
load_dotenv()

# Set the title for the Streamlit app
st.title("Knowledgeable Smart AI Agent")

# Ensure event loop exists in Streamlit's ScriptRunner thread
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Allow nested event loops (important for Streamlit)
nest_asyncio.apply()

# Cached function to create a vectordb for the provided PDF files
@st.cache_resource
def create_vectordb(files, filenames):
    with st.spinner("Building vector database..."):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, os.environ["GOOGLE_API_KEY"]
        )
    return vectordb

# Upload PDF files
pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

# Prompt template
prompt_template = """
You are a helpful Assistant who provides both PDF-based answers and general knowledge information.
When the question asked is not about pdf content, you should state that it is general information not obtained from the pdf and not professional advice.

For questions about the PDF content:
- Prioritize information from the PDF
- Include filename and page number in citations
- Keep answers factual and to the point
- Carefully use metadata (filename, page) when citing — add them at the end of each cited sentence.
- If information spans multiple pages, cite all relevant pages.

For general knowledge questions:
- Provide basic, factual information from publicly available sources
- Use simple language and brief explanations in 2-3 sentences unless more detail are requested
- Include common examples where appropriate
- Always add a disclaimer that this is general information, not professional advice

If the question is about taxes, investments, finance or retirement:
- Explain basic concepts and common options in 2-3 sentences unless more detail are requested
- Use simple examples
- Add a disclaimer about consulting professionals for personal advice 

Always respond in a friendly and approachable manner.

The PDF content is:
{pdf_extract}
"""

# Initialize prompt history
if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": "none"}]

# Show past chat
for message in st.session_state["prompt"]:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get user input
question = st.chat_input("Ask anything")

if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("Please upload a PDF first.")
        st.stop()

    # Retrieve context from vectordb
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Build system prompt with PDF context
    system_prompt = prompt_template.format(pdf_extract=pdf_extract)
    st.session_state["prompt"][0] = {"role": "system", "content": system_prompt}

    # Add user message
    st.session_state["prompt"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Prepare assistant message
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    response = llm.invoke(st.session_state["prompt"])
    botmsg.write(response.content)

    # Save assistant reply
    st.session_state["prompt"].append({"role": "assistant", "content": response.content})
