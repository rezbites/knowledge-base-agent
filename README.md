# AI-Powered Wealth Assistant

## Live Demo

Try the deployed AI-Powered Wealth Assistant here:  
[https://ai-powered-wealth-assistant.streamlit.app/](https://ai-powered-wealth-assistant.streamlit.app/)

## Overview

This project implements an AI-powered chatbot to assist users with personal finance queries, leveraging Retrieval-Augmented Generation (RAG) and large language models. The chatbot answers questions based on a curated set of finance knowledge documents (PDFs) combined with the general knowledge of the Google Gemini LLM.

---

## Approach

- **Retrieval-Augmented Generation (RAG):**  
  We use the LangChain framework with a FAISS vector store to embed and index PDF documents containing finance knowledge. Upon user queries, the system retrieves semantically relevant document chunks which are then combined with the query to provide context-aware responses.

- **Large Language Model:**  
  Google Gemini ("gemini-1.5-flash") is used as the LLM for generating responses. The model receives augmented prompts including both retrieved document contexts and the user’s question.

- **Chat Interface:**  
  A Streamlit web application serves as the frontend, providing a conversational UI where users upload PDFs and ask finance-related queries.

- **Deployment:**  
  The app can be deployed on cloud platforms such as Streamlit Cloud or Render, with environment variables securely managing API keys.

---

## Tech Stack

- **Programming Language:** Python  
- **Web Framework:** Streamlit  
- **Vector Database:** FAISS via langchain-community  
- **LLM Integration:** langchain-google-genai, Google Gemini API  
- **PDF Processing:** pypdf  
- **Environment Management:** dotenv  
- **Async Handling:** asyncio, nest_asyncio  

---

## Setup Instructions

1. **Clone the repository:**

git clone https://github.com/rezbites/ai-powered-wealth-assistant.git
cd ai-powered-wealth-assistant


2. **Create a virtual environment:**

python -m venv venv
source venv/bin/activate # On macOS/Linux
venv\Scripts\activate # On Windows


3. **Install dependencies:**

pip install -r requirements.txt


4. **Configure environment variables:**

- Copy `.env.example` to `.env`:

  ```
  cp .env.example .env
  ```

- Add your Google API key into `.env`:

  ```
  GOOGLE_API_KEY=your_actual_google_gemini_api_key_here
  ```

---

## Running the Project Locally

streamlit run app.py


Open the URL provided by Streamlit to interact with the chatbot. Upload your finance PDF documents and start asking questions.

---

## Deployment

- The app can be deployed on platforms like **Streamlit Cloud** or **Render**.
- Make sure to add your API key securely via the platform’s secrets or environment variables management.
- Start command example for Render:

streamlit run app.py --server.port $PORT --server.address 0.0.0.0


---

## Assumptions

- The user provides relevant PDFs covering the finance topics for information grounding.
- The chatbot prioritizes answering questions based on these documents when applicable.
- General finance knowledge answers will be provided cautiously with disclaimers.
- The Google Gemini API key has the necessary quota and permissions enabled.

---

## Notes

- The chatbot is designed for educational purposes only and does not replace professional financial advice.
- Long conversations and very large document sets may encounter API token limits.
- Prompts are carefully constructed to include citations of PDF metadata (filename/page number).
- Vector DB is cached to improve performance during a session.

---

## Contact

For questions or issues, reach out via the GitHub repository.

---

Thank you for trying out the AI-Powered Wealth Assistant!

