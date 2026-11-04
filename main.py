import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from ingest import (
    BASE_DIR,
    DOCUMENTS_DIR,
    get_collection,
    ingest_documents,
    retrieve_context,
    delete_document_chunks,
    clear_all_chunks,
)

# Load environment variables from project folder
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

st.set_page_config(
    page_title="Indala College | Admission Assistant",
    page_icon="🎓",
    layout="centered"
)

SYSTEM_PROMPT = """
You are the admission assistance chatbot for Indala College.

Answer using only the supplied college document excerpts.

If the information is unavailable in the excerpts, clearly say:
"Sorry, this information is not available in the supplied college documents.
Please contact the Indala College Admission Office for accurate information."

Do not invent fees, eligibility rules, admission dates, course details
or contact information.

Do not use outside knowledge or assumptions.
Use simple, polite, student-friendly language.
If the excerpts do not support an answer, say the information is unavailable.
Do not create or guess page numbers or document names.
"""

MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))
TOP_K = int(os.getenv("TOP_K", "5"))

UNAVAILABLE = (
    "Sorry, this information is not available in the supplied college "
    "documents. Please contact the Indala College Admission Office "
    "for accurate information."
)


def generate_answer(question, matches):
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Please add your Groq API key to the .env file."
        )

    context_parts = []
    for item in matches:
        context_parts.append(
            f"Source: {item['source']} | Page: {item['page']}\n"
            f"Excerpt: {item['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Supplied college document excerpts:\n{context}\n\n"
                    "Answer only if the excerpts contain enough evidence. "
                    "Do not use external knowledge."
                )
            }
        ]
    )
    return response.choices[0].message.content.strip()


def save_uploaded_files(uploaded_files):
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            continue
        destination = DOCUMENTS_DIR / Path(uploaded_file.name).name
        destination.write_bytes(uploaded_file.getbuffer())
        saved.append(destination)

    return saved


def get_saved_pdfs():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path for path in DOCUMENTS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


# Main page
st.title("🎓 Indala College")
st.subheader("AI Admission Assistant")
st.write(
    "Ask questions about courses, eligibility, fees, admission procedures "
    "and required documents. Answers are grounded in the college PDFs."
)

# Sidebar - document administration
with st.sidebar:
    st.header("Document administration")
    st.caption("Upload official college PDF documents, then process them.")

    uploaded_files = st.file_uploader(
        "Upload college PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Save uploaded PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one PDF.")
        else:
            saved = save_uploaded_files(uploaded_files)
            st.success(f"Saved {len(saved)} PDF file(s) in the documents folder.")
            st.rerun()

    if st.button("Process documents", type="primary", use_container_width=True):
        try:
            with st.spinner("Extracting, chunking and embedding PDF text..."):
                result = ingest_documents()

            st.success(
                f"Processing complete: {result['stored_chunks']} chunks stored. "
                f"ChromaDB now contains {result['collection_count']} chunks."
            )
            if result["skipped_files"]:
                st.warning(
                    "No selectable text found in: "
                    + ", ".join(result["skipped_files"])
                    + ". These may be scanned PDFs and need OCR."
                )
        except Exception as exc:
            st.error(f"Processing failed: {exc}")

    st.divider()

    # Previously saved PDFs and individual deletion
    st.subheader("Saved PDF documents")
    saved_pdfs = get_saved_pdfs()

    if saved_pdfs:
        st.caption(f"{len(saved_pdfs)} PDF file(s) saved locally.")

        selected_pdf = st.selectbox(
            "Select a PDF to delete",
            options=[path.name for path in saved_pdfs],
            key="pdf_to_delete"
        )

        if st.button("Delete selected PDF", use_container_width=True):
            try:
                # Remove indexed chunks first, then remove the PDF file.
                delete_document_chunks(selected_pdf)
                target = DOCUMENTS_DIR / Path(selected_pdf).name
                if target.exists() and target.is_file():
                    target.unlink()
                st.success(f"Deleted '{selected_pdf}' and its indexed chunks.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete '{selected_pdf}': {exc}")

        st.divider()
        st.subheader("Clear all documents")
        st.warning(
            "This permanently deletes every saved PDF in the documents folder "
            "and every indexed chunk in this admission collection."
        )
        confirm_clear = st.checkbox(
            "I understand. Delete all PDFs and indexed data.",
            key="confirm_clear_all"
        )

        if st.button(
            "Clear all documents",
            use_container_width=True,
            disabled=not confirm_clear
        ):
            try:
                # Clear the collection, including any orphaned indexed chunks.
                remaining_chunks = clear_all_chunks()

                deleted_files = 0
                for pdf_path in get_saved_pdfs():
                    pdf_path.unlink()
                    deleted_files += 1

                st.success(
                    f"Deleted {deleted_files} PDF file(s). "
                    f"Remaining indexed chunks: {remaining_chunks}."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not clear all documents: {exc}")
    else:
        st.info("No saved PDF documents found.")

    st.divider()
    st.subheader("Database status")
    try:
        st.metric("Stored text chunks", get_collection().count())
    except Exception:
        st.metric("Stored text chunks", 0)

    st.caption(
        "Only upload official documents that the college is authorized to share."
    )

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View source documents"):
                for source in message["sources"]:
                    st.write(f"**{source['source']}** — Page {source['page']}")

# User question and answer
question = st.chat_input("Ask an admission-related question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching official college documents..."):
                matches = retrieve_context(question, top_k=TOP_K)

            relevant_matches = [
                item for item in matches
                if item["distance"] <= MAX_DISTANCE
            ]

            if not relevant_matches:
                answer = UNAVAILABLE
                sources = []
            else:
                with st.spinner("Generating answer using Groq AI..."):
                    answer = generate_answer(question, relevant_matches)

                unique_sources = []
                seen = set()
                for item in relevant_matches:
                    key = (item["source"], item["page"])
                    if key not in seen:
                        seen.add(key)
                        unique_sources.append({
                            "source": item["source"],
                            "page": item["page"]
                        })
                sources = unique_sources

            st.markdown(answer)
            if sources:
                with st.expander("View source documents"):
                    for source in sources:
                        st.write(f"**{source['source']}** — Page {source['page']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as exc:
            error_message = f"Unable to generate an answer: {exc}"
            st.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })

st.caption(
    "For official confirmation of admission details, "
    "please contact the Indala College Admission Office."
)
