from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag_engine.app.processing import (
    DocumentProcessingError,
    DocumentProcessor,
)
from src.rag_engine.app.question import (
    DocumentQuestionService,
    QuestionServiceError,
)
from src.rag_engine.app.upload import (
    DocumentUploadService,
    UploadValidationError,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"


app = FastAPI(
    title="Enterprise Research Intelligence",
    version="0.4.0",
    description=(
        "Document upload, processing, indexing, "
        "question answering, and citation-aware research UI."
    ),
)


upload_service = DocumentUploadService(
    )


document_processor = DocumentProcessor(
    storage=upload_service.storage,
)


question_service = DocumentQuestionService(
    storage=upload_service.storage,
)


class AskQuestionRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    """
    Public citation contract returned by the question API.

    The frontend renders these fields without inventing source
    information.
    """

    citation_id: str
    chunk_id: str
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class GroundingResponse(BaseModel):
    valid: bool
    claim_count: int
    supported_claim_count: int
    grounding_score: float
    uncited_claims: list[str] = Field(
        default_factory=list
    )
    unsupported_claims: list[str] = Field(
        default_factory=list
    )
    invalid_citations: list[str] = Field(
        default_factory=list
    )


class AskQuestionResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    citations: list[CitationResponse]
    confidence: float
    success: bool
    failure_count: int
    retrieved_count: int
    context_count: int
    grounding: GroundingResponse | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get(
    "/",
    include_in_schema=False,
)
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "enterprise-research-intelligence",
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    content = await file.read()

    try:
        record = upload_service.upload(
            filename=file.filename,
            content=content,
        )
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return JSONResponse(
        status_code=201,
        content=asdict(record),
    )


@app.get("/documents/{document_id}")
def get_document(
    document_id: str,
) -> JSONResponse:
    try:
        record = (
            upload_service.storage.load_record(
                document_id
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return JSONResponse(
        status_code=200,
        content=asdict(record),
    )


@app.post("/documents/{document_id}/process")
def process_document(
    document_id: str,
) -> JSONResponse:
    try:
        result = document_processor.process(
            document_id
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except DocumentProcessingError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return JSONResponse(
        status_code=200,
        content=asdict(result),
    )


@app.post(
    "/documents/{document_id}/ask",
    response_model=AskQuestionResponse,
)
def ask_question(
    document_id: str,
    request: AskQuestionRequest,
) -> AskQuestionResponse:
    try:
        result = question_service.ask(
            document_id=document_id,
            question=request.question,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except QuestionServiceError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return AskQuestionResponse.model_validate(
        result
    )
