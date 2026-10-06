"""Read-only knowledge inspection endpoints, scoped to the requesting application."""

from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Path as ApiPath, Query, Request

from app.core.session_ids import SESSION_ID_MAX_LENGTH, SESSION_ID_PATTERN
from app.knowledge import (
    BeliefDetailDTO,
    BeliefListResponse,
    BeliefRecordStatus,
    ContextPreviewResponse,
    EffectiveBeliefsResponse,
    KnowledgeService,
    SavedMemoryListResponse,
)
from app.knowledge.models import BeliefFiltersDTO
from app.beliefs.models import EpistemicStatus, VisibilityPolicy

router = APIRouter()

SessionIdQuery = Annotated[
    str,
    Query(
        min_length=1,
        max_length=SESSION_ID_MAX_LENGTH,
        pattern=SESSION_ID_PATTERN,
    ),
]


def _knowledge_service(
    *,
    require_beliefs: bool = True,
    require_memories: bool = False,
    application: FastAPI,
) -> KnowledgeService:
    orchestrator = application.state.orchestrator
    repository = getattr(orchestrator, "belief_repository", None)
    provider = getattr(orchestrator, "belief_context_provider", None)
    memory_retriever = getattr(orchestrator, "memory_retriever", None)
    memory_store = getattr(memory_retriever, "memory", None)
    if require_beliefs and (repository is None or provider is None):
        raise HTTPException(status_code=503, detail="Knowledge subsystem is unavailable")
    if require_memories and memory_store is None:
        raise HTTPException(status_code=503, detail="Saved memory storage is unavailable")
    return KnowledgeService(
        owner_agent_id=orchestrator.agent_id,
        repository=repository,
        context_provider=provider,
        history_store=orchestrator.history,
        memory_store=memory_store,
    )


def _require_known_session(service: KnowledgeService, session_id: str) -> None:
    if not service.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/api/knowledge/memories", response_model=SavedMemoryListResponse)
async def list_saved_memories(request: Request):
    return _knowledge_service(
        require_beliefs=False,
        require_memories=True,
        application=request.app,
    ).list_saved_memories()


@router.get(
    "/api/knowledge/beliefs/effective",
    response_model=EffectiveBeliefsResponse,
)
async def get_effective_beliefs(session_id: SessionIdQuery, request: Request):
    service = _knowledge_service(application=request.app)
    _require_known_session(service, session_id)
    return service.effective_beliefs(session_id)


@router.get("/api/knowledge/beliefs", response_model=BeliefListResponse)
async def list_beliefs_for_inspection(
    request: Request,
    subject_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    source_sender_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    predicate: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    epistemic_status: EpistemicStatus | None = None,
    visibility: VisibilityPolicy | None = None,
    record_status: BeliefRecordStatus | None = None,
    scope_session_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    source_session_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=100000)] = 0,
):
    filters = BeliefFiltersDTO(
        subject_id=subject_id,
        source_sender_id=source_sender_id,
        predicate=predicate,
        epistemic_status=epistemic_status,
        visibility=visibility,
        record_status=record_status,
        scope_session_id=scope_session_id,
        source_session_id=source_session_id,
    )
    return _knowledge_service(application=request.app).list_beliefs(
        filters=filters,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/api/knowledge/beliefs/{belief_id}",
    response_model=BeliefDetailDTO,
)
async def get_belief_for_inspection(
    belief_id: Annotated[
        str,
        ApiPath(min_length=1, max_length=64, pattern=r"^[^\x00-\x1f\x7f]+$"),
    ],
    request: Request,
):
    detail = _knowledge_service(application=request.app).get_belief_detail(belief_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Belief not found")
    return detail


@router.get(
    "/api/knowledge/belief-context",
    response_model=ContextPreviewResponse,
)
async def get_belief_context_preview(session_id: SessionIdQuery, request: Request):
    service = _knowledge_service(application=request.app)
    _require_known_session(service, session_id)
    return service.context_preview(session_id)
