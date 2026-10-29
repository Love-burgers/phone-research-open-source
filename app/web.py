"""Standalone homepage conversation API; no comparison routes or imports."""
from pathlib import Path
import logging
import sqlite3
import ipaddress
import secrets
import re
import hashlib
from urllib.parse import urlsplit
from pydantic import SecretStr
from time import monotonic
from fastapi import FastAPI, Path as ApiPath
from fastapi.responses import FileResponse, JSONResponse
from .config import APP_NAME, get_model_api_key, get_settings
from .chat import get_chat_service
from .chat_schemas import ChatRequest, ChatSession, SESSION_PATTERN
from .chat_store import SessionConflict, SessionNotFound
from .usage_guard import QuotaExceeded, get_usage_guard
from .model_credentials import use_model_api_key
STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title=APP_NAME, description="多轮选机对话 Agent", version="0.4.0")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

@app.middleware("http")
async def access_and_quota(request, call_next):
    model_key=None
    if request.url.path.startswith('/api/chat'):
        settings = get_settings()
        required = settings.chat_access_token
        host = request.client.host if request.client else ''
        try:
            remote = not ipaddress.ip_address(host).is_loopback
        except ValueError:
            remote = False  # Non-IP transports (Unix sockets/test transports).
        scheme,_,credential=request.headers.get('authorization','').partition(' ')
        supplied=credential if scheme.lower()=='bearer' else ''
        if required and not secrets.compare_digest(supplied.encode(),required.get_secret_value().encode()):
            return JSONResponse(status_code=401,content={'detail':'请输入访问密钥后继续。'})
        if remote and not required:
            return JSONResponse(status_code=403,content={'detail':'当前服务仅允许本机访问；远程访问需配置访问密钥。'})
        if request.method == 'POST' and request.url.path == '/api/chat':
            supplied_key=request.headers.get('x-model-api-key')
            if supplied_key is not None:
                if not re.fullmatch(r'[\x21-\x7e]{12,2048}',supplied_key):
                    return JSONResponse(status_code=400,content={'detail':'API Key格式无效，请检查后重试。'})
                hostname=request.url.hostname or ''
                try:
                    local_origin=ipaddress.ip_address(hostname).is_loopback
                except ValueError:
                    local_origin=hostname.lower()=='localhost'
                if request.url.scheme!='https' and (remote or not local_origin):
                    return JSONResponse(status_code=400,content={'detail':'输入个人API Key需要HTTPS连接，或在本机localhost/127.0.0.1使用。'})
                provider=urlsplit(settings.model_api_url)
                try:
                    local_provider=ipaddress.ip_address(provider.hostname or '').is_loopback
                except ValueError:
                    local_provider=provider.hostname=='localhost'
                if not (provider.scheme=='https' and provider.hostname or provider.scheme=='http' and local_provider):
                    return JSONResponse(status_code=400,content={'detail':'当前模型地址不支持安全传递个人Key，请配置HTTPS或本机模型地址。'})
                model_key=SecretStr(supplied_key)
            try:
                get_usage_guard(settings).admit(host,settings)
            except QuotaExceeded as error:
                return quota_response(error)
            except (OSError,sqlite3.DatabaseError) as error:
                return await service_unavailable(request,error)
    with use_model_api_key(model_key):
        return await call_next(request)


def quota_response(error):
    return JSONResponse(status_code=429,content={'detail':'请求过于频繁或今日用量已达上限，请稍后再试。','reason':error.reason},
                        headers={'Retry-After':str(error.retry_after)})


@app.exception_handler(QuotaExceeded)
async def quota_exceeded(_request, error):
    return quota_response(error)

@app.middleware("http")
async def request_timing(request, call_next):
    started = monotonic()
    response = await call_next(request)
    # Route templates exclude session IDs, query strings and conversation text.
    route = getattr(request.scope.get("route"), "path", "unmatched")
    logger.info("request method=%s route=%s status=%d elapsed_seconds=%.4f",
                request.method, route, response.status_code, monotonic()-started)
    return response

@app.exception_handler(SessionNotFound)
async def session_not_found(_request, _error):
    return JSONResponse(status_code=404, content={"detail":"这段对话不存在或已删除，请新建对话"})

@app.exception_handler(SessionConflict)
async def session_conflict(_request, _error):
    return JSONResponse(status_code=409, content={"detail":"这段对话已收到另一条消息，请刷新后再发送"})

async def service_unavailable(_request, _error):
    logger.warning("service_unavailable error_type=%s", type(_error).__name__)
    return JSONResponse(status_code=503, content={"detail":"选机资料或会话暂不可用，请稍后重试"})

for error_type in (OSError, ValueError, sqlite3.DatabaseError):
    app.add_exception_handler(error_type, service_unavailable)

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

@app.post("/api/chat/sessions", response_model=ChatSession)
def create_chat_session():
    return get_chat_service().store.create()

@app.get("/api/chat/sessions/{session_id}", response_model=ChatSession)
def get_chat_session(session_id: str = ApiPath(pattern=SESSION_PATTERN)):
    return get_chat_service().store.load(session_id)

@app.delete("/api/chat/sessions/{session_id}", status_code=204)
def delete_chat_session(session_id: str = ApiPath(pattern=SESSION_PATTERN)):
    get_chat_service().store.delete(session_id)

@app.post("/api/chat", response_model=ChatSession)
def chat(request: ChatRequest):
    return get_chat_service().turn(request)

@app.get("/api/health")
def health():
    settings = get_settings()
    path = settings.chat_knowledge_index_path
    service = get_chat_service()
    knowledge = service.knowledge
    return {"status": "ok", "model_configured": bool(get_model_api_key()),
            'model_name':settings.model_name,'model_api_host':urlsplit(settings.model_api_url).hostname,
            "authentication_required":bool(settings.chat_access_token),
            "chat_agent_enabled": service.agent_enabled,
            "knowledge_configured": bool(path and Path(path).is_file()),
            "knowledge_models":len({c.phone_name for c in knowledge.index.chunks}),
            "knowledge_chunks":len(knowledge.index.chunks),
            "knowledge_coverage":knowledge.coverage(),
            "knowledge_source":"configured_index" if path else "bundled_index",
            "knowledge_index_sha256":hashlib.sha256(knowledge.path.read_bytes()).hexdigest() if knowledge.path else None,
            "retrieval_method":knowledge.search_method or knowledge.index.method,
            "embedding_provider":settings.embedding_provider or None}
