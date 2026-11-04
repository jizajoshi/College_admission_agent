# Indala College – RAG-Based AI Admission Agent

## Project files

- `main.py` – Streamlit chatbot, PDF upload, retrieval and answer display.
- `ingest.py` – PDF text extraction, chunking, embeddings and ChromaDB storage.
- `documents/` – Put the college's official PDF files here.
- `database/` – ChromaDB persistent database (created automatically).
- `.env.example` – Example API configuration.
- `requirements.txt` – Python dependencies.

## 1. Open the project in VS Code

Open the `College_Admission_Agent_Starter` folder in VS Code.

## 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Add official college PDFs

Copy the college's PDF files into the `documents` folder. The application also
allows PDFs to be uploaded from its sidebar.

PDFs must contain selectable text for this starter version. Scanned image-only
PDFs need OCR; this version reports files from which no text could be extracted.

## 5. Configure the LLM

Copy `.env.example` to a new file named `.env` in the project root.
Add your own OpenAI API key:

```env
OPENAI_API_KEY=your_actual_key
OPENAI_MODEL=gpt-4o-mini
TOP_K=5
MAX_DISTANCE=0.75
```

Do not share your `.env` file or commit it to Git.

## 6. Run the application

```powershell
streamlit run main.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## 7. Demonstration

1. Upload official college PDFs using the sidebar, or copy them into `documents`.
2. Click **Save uploaded PDFs** if you uploaded files through the interface.
3. Click **Process documents**.
4. Ask at least three questions whose answers are present in the documents:
   - Which engineering courses are offered?
   - What are the eligibility criteria for Computer Engineering?
   - Which documents are required for admission?
5. Check **View source documents** below each answer.
6. Ask an unsupported question, for example:
   - Does the college provide free international student accommodation?
7. Confirm that the chatbot says the information is unavailable instead of inventing an answer.

Use questions whose answers really exist in your official PDFs. Do not present
sample or demonstration data as official college information.

## Important implementation notes

- Each chunk stores its PDF filename and 1-based page number.
- The vector database uses cosine distance; lower distance means a closer match.
- `MAX_DISTANCE=0.75` is only an initial heuristic. Test relevant and irrelevant
  questions against your own documents and tune this value.
- Retrieval relevance checks reduce unsupported answers but cannot guarantee
  perfect grounding. Review the retrieved excerpts and test the system.
- This starter version uses OpenAI for answer generation. An API key and API
  access are required. Embeddings are generated locally using Sentence Transformers.
- If your PDFs are scanned, add OCR (for example Tesseract) before relying on
  them for admission answers.
