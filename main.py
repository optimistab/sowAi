from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tempfile
import os
import weaviate
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate as WeaviateVectorStore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

client = weaviate.Client(
    url="https://f1le5xfsbapwnseyxohxg.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_client_secret=weaviate.AuthApiKey(
        api_key=""
    ),
)

def process_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        return []
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def store_chunks(chunks):
    WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        index_name="CustomerRequirements",
        text_key="text"
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    chunks = process_document(tmp_path)
    if not chunks:
        return {"message": "Unsupported file format or empty document."}

    store_chunks(chunks)
    os.remove(tmp_path)
    return {"message": f"Uploaded and stored {len(chunks)} requirement chunks."}

@app.post("/query")
async def query_requirement(payload: dict):
    query_text = payload.get("query", "")
    if not query_text:
        return {"results": []}

    # Embed the query manually
    query_vector = embedding_model.embed_query(query_text)

    vectorstore = WeaviateVectorStore(
        client=client,
        embedding=embedding_model,
        index_name="CustomerRequirements",
        text_key="text"
    )

    results = vectorstore.similarity_search_by_vector(query_vector, k=5)
    return {"results": [doc.page_content for doc in results]}

@app.get("/")
def health():
    return {"status": "OK"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
