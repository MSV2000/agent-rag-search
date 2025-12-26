import aiofiles
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
class UserRequest:
    question: str


@dataclass
class UserResponse:
    answer: str


@app.post("/add_pdf_to_db")
async def add_pdf_to_db(collection_name: str = Form(..., description="Название коллекции"),
                        start_page: int = Form(1, description="Страница, с которой начать обработку (по умолчанию 1)"),
                        overwrite: bool = Form(False, description="Перезаписать коллекцию если существует (по умолчанию False)"),
                        file: UploadFile = File(..., description="PDF файл для загрузки")):

    if not file:
        raise HTTPException(status_code=400, detail="Файл не передан")

    filename = file.filename

    # Проверка расширения файла (опционально)
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail=f"Файл {filename} не имеет расширения .pdf")

    file_path = UPLOAD_DIR / filename

    # Асинхронное сохранение файла
    try:
        async with aiofiles.open(file_path, 'wb') as buffer:
            content = await file.read()
            await buffer.write(content)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Файл {filename} не загружен: {e}")

    try:
        pdf_db.add_pdf_to_db(file_path=str(file_path), collection_name=collection_name, text_splitter=splitter,
                             start_page=start_page, overwrite=overwrite)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка на этапе извлечения текста из PDF файла и добавления в "
                                                    f"векторную базу данных: {e}")

    if overwrite:
        return {
            "message": f"Коллекция {collection_name} перезаписана. Файл {filename} загружен в коллекцию",
            "collection_name": collection_name,
            "filename": filename,
            "status": "success",
            "action": "overwritten"
        }
    return {
        "message": f"Файл {filename} загружен в коллекцию {collection_name}",
        "collection_name": collection_name,
        "filename": filename,
        "status": "success",
        "action": "added"
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
