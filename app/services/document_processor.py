import io
import pandas as pd
import fitz  # PyMuPDF
from fastapi import UploadFile

class DocumentProcessor:
    @staticmethod
    async def process_pdf(file: UploadFile) -> str:
        """Extracts text from PDF files using PyMuPDF (fitz)"""
        content = await file.read()
        
        # Load PDF from memory baytes
        doc = fitz.open(stream=content, filetype="pdf")
        text_pages = []
        for page in doc:
            text_pages.append(page.get_text())
            
        return "\n\n--- PAGE BREAK ---\n\n".join(text_pages)

    @staticmethod
    async def process_csv(file: UploadFile) -> str:
        """Loads a CSV into pandas and extracts a markdown summary/head to pass to the LLM"""
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        summary = []
        summary.append(f"CSV Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        summary.append(f"Columns: {', '.join(df.columns)}")
        summary.append("\nData Types:")
        summary.append(df.dtypes.to_string())
        
        summary.append("\nSample Data (First 5 rows):")
        summary.append(df.head().to_markdown())
        
        return "\n".join(summary)
        
    @classmethod
    async def process_file(cls, file: UploadFile) -> str:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            return await cls.process_pdf(file)
        elif filename.endswith(".csv"):
             return await cls.process_csv(file)
        else:
            # Fallback for plain text
            content = await file.read()
            return content.decode('utf-8', errors='ignore')
