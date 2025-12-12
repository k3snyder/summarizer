from pydantic import BaseModel
from typing import List, Dict, Any

class PDFDocument(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    metadata: Dict[str, Any]

class ParserOutput(BaseModel):
    chunk_id: str
    doc_title: str
    text: str
    tables: List[Dict[str, Any]]
    image_base64: List[str]
