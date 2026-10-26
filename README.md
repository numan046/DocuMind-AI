<div align="center">

# 📄 DocuMind AI

### Source-Grounded Multi-PDF Question Answering Assistant

Upload one or more PDFs, ask questions in plain language, and get answers backed by the **document name, page number and original text** of every source.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3%2070B-F55036)
![scikit-learn](https://img.shields.io/badge/Retrieval-TF--IDF-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

</div>

---

## 📖 Overview

**DocuMind AI** is a web application built with Python and Streamlit that lets a user chat with their PDF files. The app reads the uploaded documents, finds the passages that best match a question, and sends **only those passages** to an AI model. The model writes the answer from that evidence, and the app shows where each part came from. This keeps answers accurate and checkable instead of guessed.

> **Internship task:** Develop an AI-powered application that allows users to upload a PDF document and ask questions related to its content, generating accurate answers based on the information available within the PDF.
>
> **Level:** Advanced Level – Project 1 · **Domain:** Artificial Intelligence · **Organisation:** CyberSNova

---

## ✨ Features

| Feature | Description |
|---|---|
| 📚 **Multi-PDF upload** | Up to 10 PDFs at once (25 MB per file, 80 MB total, 300 pages per file) |
| 🔎 **Evidence-based retrieval** | TF-IDF search with query expansion finds the most relevant passages |
| 📌 **Source citations** | Every answer shows document name, PDF page number and the original text |
| 💬 **Conversational Q&A** | Follow-up questions use the recent chat history |
| 📝 **Document summary** | One-click summary of all uploaded documents |
| ✨ **Suggested questions** | AI-generated questions to help you explore the documents |
| 🔍 **Page inspector** | Pick a document and page to read the extracted text directly |
| 🛡️ **Offline fallback** | If the AI service fails, the app still returns a local evidence-based answer |
| 🔁 **Model fallback** | Automatically tries other Groq models if the preferred one is unavailable |
| 📤 **Export** | Download the summary and the full conversation as Markdown |
| ⚙️ **Adjustable settings** | Chunk size, chunk overlap and number of evidence passages |

---

## 🧠 How It Works

```text
PDF files ─► Text extraction (PyMuPDF) ─► Chunking (with overlap)
                                              │
                                              ▼
Question ─► Query expansion ─► TF-IDF + cosine similarity ─► Top-K passages
                                                                  │
                                                                  ▼
                                      Groq LLM (answers only from the evidence)
                                                                  │
                                                                  ▼
                             Answer + document name + page number + source text
```

1. **Extract** – PyMuPDF reads text and page layout from each PDF.
2. **Chunk** – Text is split into overlapping chunks, each tagged with its document and page numbers.
3. **Retrieve** – A TF-IDF index (scikit-learn) ranks chunks by similarity to the question.
4. **Generate** – The top passages are sent to the Groq model, which must answer from that evidence.
5. **Cite** – The app validates and displays the sources used for the answer.

---

## 🛠️ Tech Stack

| Tool | Used for |
|---|---|
| **Python 3** | Main programming language |
| **Streamlit** | Web interface: upload, chat box, tabs, sidebar settings |
| **PyMuPDF** | Reading text and metadata from PDF files |
| **scikit-learn (TF-IDF)** | Searching the PDF text for matching passages |
| **NumPy** | Combining and ranking search scores |
| **Groq API (Llama 3.3 70B)** | AI model for answers, summaries and suggested questions |
| **OpenAI Python library** | Client used to call Groq's OpenAI-compatible API |

---

## 📁 Project Structure

```text
DocuMind-AI/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .streamlit/
    └── secrets.toml     # (optional, not committed) API key
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create a virtual environment *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```text
streamlit
openai
pymupdf
numpy
scikit-learn
```

### 4. Set your Groq API key

Choose **one** of the following:

**Option A – Environment variable**
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_api_key_here"

# macOS / Linux
export GROQ_API_KEY="your_api_key_here"
```

**Option B – Streamlit secrets** (`.streamlit/secrets.toml`)
```toml
GROQ_API_KEY = "your_api_key_here"
```

### 5. Run the app
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501**.

---

## 🖥️ Usage

1. Upload one or more PDF files from the sidebar.
2. Wait for the documents to be processed.
3. Open the **💬 Ask the PDFs** tab and type your question.
4. Read the answer and expand the sources to see the exact page and text.
5. Use **📝 Summary**, **✨ Suggested Questions** and **📤 Export** for more.

**Sidebar settings**

| Setting | Range | Default |
|---|---|---|
| Chunk size (chars) | 500 – 1500 | 900 |
| Chunk overlap (chars) | 0 – 400 | 200 |
| Evidence passages | 3 – 10 | 6 |

---


## ⚠️ Limitations

- Works with text-based PDFs; scanned image-only PDFs need OCR first.
- Answer quality depends on the clarity of the PDF text and the chosen model.
- Free Groq usage is subject to rate limits.

---

## 🔮 Future Improvements

- OCR support for scanned PDFs
- Semantic (embedding-based) retrieval
- Support for DOCX and TXT files
- Persistent document library across sessions

---

## 👤 Author

**Muhammad Numan**
AI Engineering Intern at **[CyberSNova](https://cybersnova.co.in)**

---

<div align="center">

⭐ If you found this project useful, consider giving it a star.

</div>
