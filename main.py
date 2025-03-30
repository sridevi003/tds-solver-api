from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from typing import List, Optional
import pandas as pd
import zipfile
import io
import requests
from fastapi.middleware.cors import CORSMiddleware
import os

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AI_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

app = FastAPI()

origins = ["*"]  # Adjust as needed for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/")
async def process_request(
    question: Optional[str] = Form(None),  # Optional question
    file: Optional[UploadFile] = File(None),  # Optional file
    name: Optional[List[str]] = Query(None)   # Optional list of names
):
    if name:
        if not name:
            raise HTTPException(status_code=400, detail="At least one 'name' parameter is required")
        return {"names": name}

    if question:
        if file and file.filename.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(await file.read()), "r") as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]
                    if not csv_files:
                        raise HTTPException(status_code=400, detail="No CSV file found in ZIP.")

                    with zip_ref.open(csv_files[0]) as f:
                        df = pd.read_csv(f)
                        if "answer" in df.columns:
                            return {"answer": str(df["answer"].iloc[0])}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing ZIP file: {str(e)}")

        try:
            if not AIPROXY_TOKEN:
                raise HTTPException(status_code=500, detail="AIPROXY_TOKEN is not set.")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AIPROXY_TOKEN}"
            }
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are an expert AI assistant designed to solve graded assignment questions from the IIT Madras Data Science online degree program. Your task is to accurately answer questions based on the provided text or data. If a CSV file is given, analyze the data within it to find the answer. If no CSV file is provided, use your general knowledge and reasoning abilities to answer the question. Give only the answer to directly submit it."},
                    {"role": "user", "content": question}
                ]
            }
            response = requests.post(AI_PROXY_URL, headers=headers, json=data)
            response_json = response.json()

            if "choices" not in response_json or not response_json["choices"]:
                raise HTTPException(status_code=500, detail="Invalid response from AI Proxy API.")

            return {"answer": response_json["choices"][0]["message"]["content"].strip()}

        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"AI Proxy API error: {str(e)}")

    raise HTTPException(status_code=400, detail="Invalid request. Provide either 'name' parameters or a 'question' (optionally with a 'file').")
