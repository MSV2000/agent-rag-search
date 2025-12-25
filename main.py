import aiofiles
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, UploadFile, File
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_to_db import PDFVecDataBase


pdf_db = PDFVecDataBase(embeddings_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        path_db="chroma_db")

splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1500,
                                          separators=["\n\n", "\n", ",", " ", ""])

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()


@dataclass
class UploadRequest:
    collection_name: str
    start_page: int
    overwrite: bool
    files: List[UploadFile] = File(...)


@dataclass
class UserRequest:
    question: str


@dataclass
class UserResponse:
    answer: str


@app.post("/add_pdf_to_db")
async def add_pdf_to_db(collection_name: str, start_page: int, overwrite: bool, files: List[UploadFile] = File(...)):
    # files = data.files
    # collection_name = data.collection_name
    # start_page = data.start_page
    # overwrite = data.overwrite

    if not files:
        raise HTTPException(status_code=400, detail="Не переданы файлы")

    saved_files = []
    invalid_files = []

    for file in files:
        safe_filename = file.filename

        # Проверка расширения файла (опционально)
        if not safe_filename.lower().endswith('.pdf'):
            invalid_files.append(safe_filename)
            continue

        file_path = UPLOAD_DIR / safe_filename

        # Асинхронное сохранение файла
        try:
            async with aiofiles.open(file_path, 'wb') as buffer:
                content = await file.read()
                await buffer.write(content)

            saved_files.append(safe_filename)
        except Exception as e:
            invalid_files.append(f"{safe_filename}: {str(e)}")

    if not saved_files:
        raise HTTPException(status_code=400, detail="Нет допустимых файлов для загрузки")

    for safe_file in saved_files:
        file_path = str(UPLOAD_DIR / safe_file)

        pdf_db.add_pdf_to_db(file_path=file_path, collection_name=collection_name, text_splitter=splitter,
                             start_page=start_page, overwrite=False)

    return {
        "message": "Обработка файлов завершена",
        "saved_files": saved_files,
        "invalid_files": invalid_files,
    }


@app.post("/question")
async def answers_questions(data: UserRequest) -> UserResponse:
    try:
        question = data.question

        if not question:
            raise HTTPException(status_code=422, detail="Отсутствует вопрос")
        else:
            return UserResponse(answer="Good job!")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка выполнения запроса: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.10.148", port=8080)
