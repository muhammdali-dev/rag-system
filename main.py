from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

app = FastAPI(title="RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Multiple documents store karne ke liye
documents_store = {}

class Question(BaseModel):
    question: str
    doc_id: str

@app.get("/")
def home():
    return {"status": "running", "message": "RAG System API"}

@app.get("/ui")
def ui():
    return FileResponse("index.html")

@app.get("/documents")
def get_documents():
    return {"documents": list(documents_store.keys())}
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # File check karo
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Upload PDF Files only!")
    
    # Temp file mein save karo
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    # PDF load karo
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # Chunks banao
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    
    # Vector store banao
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Store mein save karo
    doc_id = file.filename.replace('.pdf', '')
    documents_store[doc_id] = vector_store
    
    # Temp file delete karo
    os.unlink(tmp_path)
    
    return {
        "message": f"{file.filename} Uploaded successfully!",
        "doc_id": doc_id,
        "pages": len(docs),
        "chunks": len(chunks)
    }

@app.post("/ask")
async def ask_question(q: Question):
    if q.doc_id not in documents_store:
        raise HTTPException(
            status_code=400,
            detail="Document not found! Please upload the document first."
        )
    
    vector_store = documents_store[q.doc_id]
    docs = vector_store.similarity_search(query=q.question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    from langchain_core.messages import HumanMessage
    
    prompt = f"""Answer based on this document content only:

Context:
{context}

Question: {q.question}

Answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "question": q.question,
        "answer": response.content,
        "doc_id": q.doc_id
    }

@app.delete("/document/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found!")
    
    del documents_store[doc_id]
    return {"message": f"{doc_id} deleted successfully!"}