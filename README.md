# Chatbot with Upload PDF or Text File 🧠📄

This project is a smart chatbot built using Python that allows users to upload PDF or text files and interact with the content using natural language queries. It leverages vector embeddings and Google Gemini (Generative AI) to understand document content and provide intelligent answers based on the uploaded files.

## 🧩 Features

- 📂 Upload support for `.pdf` and `.txt` files  
- 💬 Interactive chatbot interface
- 📌 Memory for previous conversations
- 📊 Supports multiple file uploads  
- 🧠 Context-aware conversation memory  
- 🔍 Semantic search with vector embeddings  
- ⚡ Fast and efficient document processing  

## 📺 Demo Video


https://github.com/user-attachments/assets/27bb8bf1-a1c0-4666-af54-f664dd7991f1



## 🛠️ Technologies Used

- **Programming Language**: Python  
- **Web Framework**: Streamlit (for the user interface)  
- **Generative AI**: Google Gemini (`ChatGoogleGenerativeAI`)  
- **Document Parsing**: PyPDF2, Textract  
- **Vector Embedding & Search**: Langchain, SentenceTransformers, FAISS  



## 🚀 Getting Started


### Prerequisites

Make sure you have **Python 3.7+** installed. Then install the required packages:

```bash
pip install -r requirements.txt
````

### Run the Application

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`) in your browser.



## 📁 Project Structure

```
Chatbot-with-Upload-PDF/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── utils/                  # Utility functions (embedding, parsing, etc.)
├── samples/                # Sample PDF/text files (optional)
└── README.md
```



## 🔐 Environment Variables

Create a `.env` file in your root directory and add the following:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
GEMINI_API_KEY=your_google_gemini_api_key_here
```

> ⚠️ **Important**: Never push your `.env` file to GitHub. Make sure to add `.env` to your `.gitignore` to keep your keys secure.



## 📦 Sample Use Cases

* Extract insights from research papers
* Upload contracts and ask specific legal or business questions
* Interact with meeting transcripts or notes



## 💡 Future Improvements

* 🔐 User authentication and session management
* 🧾 Chat export options
* 📊 Support for additional file types like `.docx` or `.xlsx`



## 📬 Contact

Created by [Arpit Padmani](https://github.com/Arpit-Padmani)
Feel free to open an issue or pull request to contribute!

---

⭐ **Star this repo** if you found it useful!





