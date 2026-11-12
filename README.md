
# VideoNotes 🎬
### AI-Powered Video Assistant

VideoNotes is an AI-powered application that transforms YouTube videos and local audio/video files into searchable, structured information.

It uses speech-to-text, Large Language Models (LLMs), embeddings and Retrieval-Augmented Generation (RAG) to help users understand video content and ask questions about it.

## ✨ Features

- **YouTube Audio Extraction:** Extract audio from YouTube videos.
- **Local File Support:** Process local audio and video files.
- **Speech Transcription:** Whisper for English and Sarvam AI for Hinglish.
- **AI Summarization:** Generate concise summaries of long video transcripts.
- **Automatic Title Generation:** Generate meaningful titles from video content.
- **Action Item Extraction:** Identify tasks, owners and deadlines.
- **Key Decision Extraction:** Identify important decisions discussed.
- **Open Question Extraction:** Find unresolved questions and follow-up topics.
- **RAG-Based Question Answering:** Ask questions and receive answers grounded in video transcripts.
- **Vector Database:** Store transcript embeddings using ChromaDB.
- **Video Library:** Save, retrieve and revisit previously analysed videos.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Speech-to-Text | OpenAI Whisper, Sarvam AI |
| LLM | Groq |
| LLM Framework | LangChain |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Audio Processing | FFmpeg, Pydub, yt-dlp |
| Environment Management | Python venv, python-dotenv |

## 🏗️ Project Architecture

```text
                User Input
                     |
          YouTube URL / Local File
                     |
                     v
             Audio Processing
                     |
              Audio Chunking
                     |
                     v
               Transcription
              /             \
          Whisper          Sarvam AI
          English           Hinglish
              \             /
                     |
                     v
                Transcript
                     |
          +----------+-----------+
          |          |           |
          v          v           v
       Summary    Information   Title
                  Extraction
                     |
          +----------+-----------+
          |                      |
          v                      v
     Video Library          Text Chunking
                                  |
                                  v
                              Embeddings
                                  |
                                  v
                               ChromaDB
                                  |
                                  v
                            RAG Retriever
                                  |
                           User Question
                                  |
                                  v
                              Groq LLM
                                  |
                                  v
                            AI Answer
```

## 📂 Project Structure

```text
VideoNotes/
│
├── app.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── core/
│   ├── transcriber.py
│   ├── summarize.py
│   ├── extractor.py
│   ├── vector_store.py
│   ├── rag_eng.py
│   └── video_library.py
│
├── utils/
│   └── audio_processor.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/NishaK23/VideoNotes.git
cd VideoNotes
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg must be installed separately and added to your system PATH.

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5

WHISPER_MODEL=small
```

Add your own API keys. Never share your actual `.env` file publicly.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use VideoNotes.

## 🔍 How RAG Works in VideoNotes

VideoNotes uses Retrieval-Augmented Generation (RAG) to answer questions about video content.

1. The transcript is divided into smaller overlapping text chunks.
2. Hugging Face generates embeddings for the chunks.
3. The embeddings and associated documents are stored in ChromaDB.
4. When a user asks a question, similarity search retrieves the four most relevant chunks.
5. The retrieved transcript context and question are passed to the Groq LLM.
6. The LLM generates an answer based on the retrieved video content.

The application instructs the LLM not to invent information when the answer is not supported by the retrieved context.

## 🚀 Future Improvements

- Timestamp-based references in answers.
- Improved retrieval and reranking.
- Faster transcription and processing.
- Structured extraction using schema-based outputs.
- Support for additional languages.
- Evaluation of transcription and RAG answer quality.

## 👩‍💻 Author

**Nisha K**

GitHub: [NishaK23](https://github.com/NishaK23)

---

⭐ If you find this project interesting, consider starring the repository!
