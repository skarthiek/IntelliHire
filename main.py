from fastapi import FastAPI, UploadFile, File, Form
import os
from dotenv import load_dotenv

load_dotenv()

from utils.pdf_parser import extract_text_from_pdf
from utils.csv_parser import extract_text_from_csv
from utils.rag_pipeline import create_vectorstore, get_answer

app = FastAPI()
db = None

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".csv"):
        text = extract_text_from_csv(file_path)
    else:
        return {"error": "Unsupported file type"}

    global db
    db = create_vectorstore(text)
    return {"message": f"{file.filename} uploaded and processed."}

@app.post("/chat/")
async def chat_with_file(query: str = Form(...)):
    if db is None:
        return {"error": "Please upload a file first."}
    
    answer = get_answer(db, query)
    return {"response": answer}
