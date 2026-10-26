DocuMind AI 📄🤖

DocuMind AI is a Streamlit-based application that lets users upload multiple PDF documents and ask questions about their content. It uses Groq-powered language models and TF-IDF retrieval to find relevant passages and generate document-grounded answers.

Features

Upload and process multiple PDFs

Ask questions about uploaded documents

Retrieve relevant text passages using TF-IDF and cosine similarity

Generate answers with page references

Summarize documents and suggest questions

Export conversations in Markdown format

Simple, interactive Streamlit interface

Tech Stack

Python

Streamlit

Groq API with the OpenAI Python SDK

PyMuPDF

scikit-learn and NumPy

Getting Started

1. Clone the repository

git clone https://github.com/numan046/DocuMind-AI.git
cd DocuMind-AI

2. Install dependencies

pip install -r requirements.txt

3. Configure your Groq API key

Set the GROQ_API_KEY environment variable, or add it to .streamlit/secrets.toml:

GROQ_API_KEY = "your_groq_api_key"

4. Run the app

streamlit run app.py

Limitations

Scanned PDFs may not work without OCR.

Retrieval is based on TF-IDF text matching, not embedding-based vector search.

Answer quality depends on the extracted PDF text and the selected language model.

Author

Muhammad Numan

GitHub: @numan046

LinkedIn: numandinmuhammad

Hugging Face: numandinmuhammad
