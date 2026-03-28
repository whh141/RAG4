"""Chat WebSocket router layer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from app.db.sqlite_mgr import SQLiteManager
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])

chat_service = ChatService(sqlite_mgr=SQLiteManager())


@router.websocket("/stream/{session_id}")
async def stream_chat(session_id: str, websocket: WebSocket) -> None:
    """Accept query payload and stream strictly-defined packets from service layer."""
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        if not isinstance(payload, dict) or not isinstance(payload.get("query"), str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="WebSocket payload must be JSON with string field 'query'.",
            )

        async for packet in chat_service.stream_chat(session_id=session_id, user_query=payload["query"]):
            await websocket.send_json(packet)
    except HTTPException as exc:
        await websocket.close(code=1008, reason=exc.detail)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc))


__all__ = ["router"]
