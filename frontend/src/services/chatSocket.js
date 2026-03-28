export function createChatSocket(sessionId) {
  const base = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'
  return new WebSocket(`${base}/api/chat/stream/${sessionId}`)
}
