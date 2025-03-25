from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import json
import hashlib
import fitz  # PyMuPDF
from docx import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  # Đọc biến môi trường từ .env

import faiss
import numpy as np

load_dotenv()

# Khởi tạo FastAPI
app = FastAPI()

# Khởi tạo embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Cấu hình thư mục lưu trữ vector store
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Khởi tạo FAISS index
dimension = 768  # Kích thước vector embedding
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(dimension)

# Load metadata nếu tồn tại
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        vector_store_data = json.load(f)
else:
    vector_store_data = []

# Hàm tính mã hash của nội dung file
def calculate_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Hàm đọc PDF và chia thành các chunk
def read_pdf_chunks(file_path: str) -> list:
    chunks = []
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks.append({"page": page_num + 1, "text": text})
    return chunks

# Hàm đọc DOCX và chia thành các chunk
def read_docx_chunks(file_path: str) -> list:
    doc = Document(file_path)
    chunks = []
    for para_num, para in enumerate(doc.paragraphs):
        if para.text.strip():
            chunks.append({"paragraph": para_num + 1, "text": para.text})
    return chunks

# Endpoint upload file
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Kiểm tra định dạng file
        if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF hoặc DOCX.")

        # Lưu file tạm thời
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Tính mã hash của file
        file_hash = calculate_hash(temp_path)

        # Kiểm tra xem file đã tồn tại chưa dựa trên mã hash
        if any(data.get("hash") == file_hash for data in vector_store_data):
            os.remove(temp_path)  # Xóa file tạm
            return {"message": "File đã tồn tại, không embedding lại.", "file": file.filename}

        # Nếu file chưa tồn tại, tiếp tục xử lý
        # Đọc và chia chunk
        if file.filename.endswith(".pdf"):
            chunks = read_pdf_chunks(temp_path)
        else:
            chunks = read_docx_chunks(temp_path)

        # Xóa file tạm sau khi đọc
        os.remove(temp_path)

        # Tạo embedding và lưu từng chunk
        for chunk in chunks:
            embedding_vector = embeddings.embed_query(chunk["text"])
            vec_np = np.array(embedding_vector, dtype=np.float32).reshape(1, -1)
            index.add(vec_np)

            # Lưu metadata, bao gồm mã hash
            vector_store_data.append({
                "filename": file.filename,
                "hash": file_hash,
                "chunk_info": {
                    "page": chunk.get("page", None),
                    "paragraph": chunk.get("paragraph", None)
                },
                "content_preview": chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"]
            })

        # Lưu FAISS index và metadata
        faiss.write_index(index, INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(vector_store_data, f, ensure_ascii=False, indent=4)

        return {"message": "Upload và lưu vector store thành công.", "file": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

# Endpoint để hỏi Gemini AI
@app.post("/ask")
async def ask_question(question: str):
    try:
        # Tạo embedding cho câu hỏi
        question_embedding = embeddings.embed_query(question)
        question_vec = np.array(question_embedding, dtype=np.float32).reshape(1, -1)

        # Tìm kiếm các chunk liên quan trong FAISS index
        D, I = index.search(question_vec, k=5)  # Tìm 5 chunk gần nhất

        # Lấy nội dung của các chunk liên quan
        related_chunks = []
        for idx in I[0]:
            if idx != -1:  # Kiểm tra chỉ số hợp lệ
                chunk_info = vector_store_data[idx]
                related_chunks.append(chunk_info["content_preview"])

        # Tạo prompt bao gồm nội dung các chunk liên quan và câu hỏi
        context = "\n".join(related_chunks)  # Nối các chunk thành một chuỗi
        # print('context', context)
        prompt = f"""
        Bạn chỉ trả lời dựa trên những thông tin trong tài liệu. 
        Nếu tài liệu không liên quan đến câu hỏi, hãy trả lời "Không biết".
        Dựa trên thông tin từ các tài liệu sau:
        {context}

        Hãy trả lời câu hỏi sau: {question}
        """

        print(prompt)
        # Gửi yêu cầu đến Gemini AI
        response = llm.invoke(prompt)
        # print('response', response.content)

        # Trả về câu trả lời
        return {"answer": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")