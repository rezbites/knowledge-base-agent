# 🧠 Knowledge Agent - AI-Powered Document Q&A

An intelligent chatbot that answers questions based on your PDF documents using RAG (Retrieval-Augmented Generation) with Google's Gemini AI.


## DEMO LINK

## (https://drive.google.com/file/d/1EZSBlBCW21_HuCRzZ8BrezrWMmp_Lfv0/view?usp=drive_link)

## DEPLOYED

## https://rag-corp-intel.streamlit.app

## ✨ Features

- 📄 **PDF Document Processing** - Upload and index multiple PDF files
- 💬 **Multi-Chat Management** - Create and manage multiple chat sessions
- 🔍 **Intelligent Search** - FAISS vector database for semantic search
- ⚡ **Real-time Streaming** - See AI responses as they're generated
- 📝 **Source Citations** - Answers include page numbers and filenames
- 🎨 **Clean UI** - Modern, intuitive interface built with Streamlit
- 🔄 **Chat History** - Rename and switch between conversations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or 3.10
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/knowledge-base-agent.git
cd knowledge-base-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Project Structure

```
knowledge-base-agent/
├── app.py              # Main Streamlit application
├── brain.py            # PDF processing and vector database logic
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── README.md          # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Your Google Gemini API key | Yes |

### Streamlit Configuration

For deployment on Streamlit Cloud, add your `GOOGLE_API_KEY` in:
- **Settings** → **Secrets** → Add as: `GOOGLE_API_KEY = "your_key_here"`

## 📖 Usage

1. **Upload Documents**
   - Click the "➕" button or wait for the initial prompt
   - Upload one or more PDF files
   - Click "Submit" to process and index the documents

2. **Ask Questions**
   - Type your question in the chat input
   - The AI will search your documents and provide answers with citations
   - Citations format: `(filename: document.pdf, page: 5)`

3. **Manage Chats**
   - Create new chats with the "➕ Create New Chat" button
   - Switch between chats by clicking on them in the sidebar
   - Rename chats by clicking the "✏️" button

## 🛠️ Technical Details

### Architecture

```
User Input → Vector Search (FAISS) → Context Retrieval → 
Gemini AI (with context) → Streaming Response → User
```

### Key Components

- **Frontend**: Streamlit
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Google Generative AI Embeddings (`text-embedding-004`)
- **LLM**: Google Gemini 2.5 Flash
- **PDF Processing**: PyPDF
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter

### Features in Detail

#### PDF Processing
- Extracts text from PDF pages
- Splits into 1000-character chunks with 100-character overlap
- Creates metadata: page numbers, chunk indices, filenames

#### Vector Search
- Converts text chunks to embeddings
- Stores in FAISS index for fast similarity search
- Retrieves top 3 most relevant chunks per query

#### AI Responses
- Uses RAG pattern: retrieves relevant context before answering
- Streams responses in real-time
- Includes automatic retry logic for rate limits
- Cites sources with page numbers

## ⚙️ Customization

### Adjust Chunk Size
In `brain.py`, modify the text splitter parameters:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Increase for more context per chunk
    chunk_overlap=100,    # Increase for better continuity
)
```

### Change AI Model
In `app.py`, switch the model:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Options: gemini-1.5-flash, gemini-1.5-pro
    temperature=0.3,            # Lower = more focused, higher = more creative
)
```

### Modify Number of Retrieved Chunks
In `app.py`:
```python
search_results = current_vectordb.similarity_search(prompt, k=3)  # Change k value
```

## 🚨 Troubleshooting

### Rate Limit Errors
If you see `429 quota exceeded`:
- **Free tier limits**: 15 requests/minute, 1,500 requests/day
- **Solution**: Wait 60 seconds or upgrade to paid plan
- The app includes automatic retry logic (3 attempts with exponential backoff)

### Import Errors
```bash
ModuleNotFoundError: No module named 'langchain_text_splitters'
```
**Solution**: Make sure you have the latest requirements installed:
```bash
pip install --upgrade -r requirements.txt
```

### PDF Processing Fails
- Ensure PDFs are text-based (not scanned images)
- For image-based PDFs, consider using OCR preprocessing
- Check file size (very large PDFs may timeout)

### Streamlit Deployment Issues
- Verify Python version is 3.9 or 3.10
- Add `.python-version` file with content: `3.10`
- Ensure `GOOGLE_API_KEY` is in Streamlit Secrets

## 📊 Performance Tips

1. **Reduce PDF size** before upload for faster processing
2. **Use smaller chunk sizes** (600-800) for faster retrieval
3. **Decrease k value** (1-2 chunks) for faster responses
4. **Enable caching** - already implemented for repeat queries
5. **Upgrade API tier** for production use

## 🔐 Security Notes

- Never commit `.env` file to version control
- Add `.env` to `.gitignore`
- Use Streamlit Secrets for deployment
- Rotate API keys regularly

## 📝 Requirements

```txt
# Core Streamlit dependencies
streamlit>=1.30.0
python-dotenv
nest-asyncio

# RAG/PDF Processing Libraries
pydantic>=2.0.0,<3.0.0
pypdf
langchain>=0.1.20
langchain-core>=0.1.52
langchain-community>=0.0.38
langchain-text-splitters>=0.0.1
langchain-google-genai>=1.0.0
faiss-cpu>=1.8.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Gemini AI](https://deepmind.google/technologies/gemini/)
- Vector search with [FAISS](https://github.com/facebookresearch/faiss)
- LangChain for [RAG implementation](https://python.langchain.com/)

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review Google Gemini API [documentation](https://ai.google.dev/gemini-api/docs)

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets: `GOOGLE_API_KEY = "your_key"`
5. Deploy!

### Deploy on Other Platforms

- **Heroku**: Add `setup.sh` and `Procfile`
- **AWS**: Use EC2 or ECS
- **Docker**: Create `Dockerfile` with Python 3.10 base image

---

Made with ❤️ using Streamlit and Google Gemini AI
