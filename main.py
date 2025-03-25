from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import json
import hashlib
import fitz  # PyMuPDF
from docx import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import faiss
import numpy as np

# Khởi tạo FastAPI
app = FastAPI()

# Thư mục lưu trữ FAISS và metadata
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Khởi tạo FAISS index
dimension = 768  # Kích thước vector của Ollama embeddings
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(dimension)

# Load metadata nếu có
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        vector_store_data = json.load(f)
else:
    vector_store_data = []


# Hàm tính hash của file
def calculate_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


# Hàm chia văn bản thành các chunk dựa trên số ký tự
def split_into_chunks(text: str, max_length: int = 1000) -> list:
    chunks = []
    current_chunk = ""

    for sentence in text.split(". "):  # Chia theo câu để giữ ngữ nghĩa
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [{"chunk_id": i + 1, "text": chunk} for i, chunk in enumerate(chunks)]


def read_pdf_chunks(file_path: str) -> list:
    full_text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            full_text += page.get_text() + " "
    return split_into_chunks(full_text)


def read_docx_chunks(file_path: str) -> list:
    full_text = ""
    doc = Document(file_path)
    for para in doc.paragraphs:
        if para.text.strip():
            full_text += para.text + " "
    return split_into_chunks(full_text)


# Khởi tạo embeddings và LLM từ Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="gemma3:12b", temperature=0)


# Endpoint upload file
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
            raise HTTPException(
                status_code=400, detail="Chỉ hỗ trợ file PDF hoặc DOCX."
            )

        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_hash = calculate_hash(temp_path)

        if any(data.get("hash") == file_hash for data in vector_store_data):
            os.remove(temp_path)
            return {
                "message": "File đã tồn tại, không embedding lại.",
                "file": file.filename,
            }

        if file.filename.endswith(".pdf"):
            chunks = read_pdf_chunks(temp_path)
        else:
            chunks = read_docx_chunks(temp_path)

        os.remove(temp_path)

        for chunk in chunks:
            embedding_vector = embeddings.embed_query(chunk["text"])
            vec_np = np.array(embedding_vector, dtype=np.float32).reshape(1, -1)
            index.add(vec_np)

            vector_store_data.append(
                {
                    "filename": file.filename,
                    "hash": file_hash,
                    "chunk_info": {"chunk_id": chunk["chunk_id"]},
                    "content_preview": (
                        chunk["text"][:500] + "..."
                        if len(chunk["text"]) > 500
                        else chunk["text"]
                    ),
                }
            )

        faiss.write_index(index, INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(vector_store_data, f, ensure_ascii=False, indent=4)

        return {
            "message": "Upload và lưu vector store thành công.",
            "file": file.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


# Endpoint để hỏi LLM
@app.post("/ask")
async def ask_question(question: str):
    try:
        question_embedding = embeddings.embed_query(question)
        question_vec = np.array(question_embedding, dtype=np.float32).reshape(1, -1)

        # Tìm kiếm top 5 tài liệu liên quan
        D, I = index.search(question_vec, k=5)

        # Chuẩn bị danh sách các tài liệu kèm score
        related_chunks = []
        for i, idx in enumerate(I[0]):
            if idx != -1:  # Kiểm tra xem chỉ số có hợp lệ không
                chunk_info = vector_store_data[idx]
                distance = D[0][i]  # Khoảng cách L2 tương ứng
                # Chuyển đổi khoảng cách thành score (càng nhỏ càng tốt, nên lấy nghịch đảo)
                score = 1 / (1 + distance)  # Công thức đơn giản để tính score (0 đến 1)
                related_chunks.append(
                    {
                        "content": chunk_info["content_preview"],
                        "score": float(
                            score
                        ),  # Chuyển sang float để JSON serialize được
                        "filename": chunk_info["filename"],
                        "chunk_id": chunk_info["chunk_info"]["chunk_id"],
                    }
                )

        # Tạo context từ nội dung các chunk
        context = "\n".join(chunk["content"] for chunk in related_chunks)

        # Tạo prompt cho LLM
        prompt = f"""
        Bạn chỉ trả lời dựa trên những thông tin trong tài liệu. 
        Nếu tài liệu không liên quan đến câu hỏi, hãy trả lời "Không biết".
        Dựa trên thông tin từ các tài liệu sau:
        {context}

        Hãy trả lời câu hỏi sau: {question}
        """

        response = llm.invoke(prompt)

        # Trả về câu trả lời và danh sách tài liệu kèm score
        return {"answer": response.content, "documents": related_chunks}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
