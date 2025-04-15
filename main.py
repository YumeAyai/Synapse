import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict
import ollama
import json
from rag_module import get_rag_context  # 导入RAG上下文生成函数

# 初始化 FastAPI
app = FastAPI()

# WebSocket管理
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_context: Dict[str, List[Dict[str, str]]] = {}
        self.user_config: Dict[str, Dict[str, float]] = {}
        self.user_quest_count: Dict[str, int] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    def get_user_context(self, user_id: str) -> List[Dict[str, str]]:
        return self.user_context.get(user_id, [])

    def update_user_context(self, user_id: str, role: str, content: str):
        if user_id not in self.user_context:
            self.user_context[user_id] = []
        self.user_context[user_id].append({"role": role, "content": content})

    def update_user_config(self, user_id: str, config: Dict[str, float]):
        self.user_config[user_id] = config

    def get_user_config(self, user_id: str) -> Dict[str, float]:
        return self.user_config.get(user_id, {"temperature": 0.7})

    def increment_user_quest_count(self, user_id: str):
        if user_id not in self.user_quest_count:
            self.user_quest_count[user_id] = 0
        self.user_quest_count[user_id] += 1

    def get_user_quest_count(self, user_id: str) -> int:
        return self.user_quest_count.get(user_id, 0)

# 创建 WebSocket 管理器
manager = WebSocketManager()

# 空的认证函数
def authenticate_user(token: str) -> bool:
    return True if token == "valid_token" else False

# WebSocket 路由：聊天消息
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    token = "valid_token"
    if not authenticate_user(token):
        await websocket.close()
        return
    await manager.connect(websocket)
    user_id = str(uuid.uuid4())
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "")
            
            if message_type == "update_config":
                config = data.get("payload", {})
                manager.update_user_config(user_id, config)
                await websocket.send_json({"type": "config_update_success", "message": "Configuration updated successfully!"})
                continue
            
            if message_type == "get_status":
                config = manager.get_user_config(user_id)
                quest_count = manager.get_user_quest_count(user_id)
                await websocket.send_json({"type": "system_status", "payload": {**config, "requests_count": quest_count}})
                continue
            
            user_message = data.get("content", "")
            if user_message:
                manager.increment_user_quest_count(user_id)
                stream_id = f"stream-{uuid.uuid4()}"
                context = manager.get_user_context(user_id)
                user_config = manager.get_user_config(user_id)
                
                await send_ollama_stream(
                    user_message, websocket, stream_id, context, user_config, user_id
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 调用 Ollama 并将响应流式传送给客户端
async def send_ollama_stream(
    user_message: str,
    websocket: WebSocket,
    stream_id: str,
    context: List[Dict[str, str]],
    user_config: Dict[str, float],
    user_id: str,
):
    try:
        # 获取RAG上下文（异步执行）
        rag_context = await asyncio.to_thread(get_rag_context, user_message)
        
        # 构建系统提示
        system_content = f"""
        你是一名严格遵守知识引用规范的生物学教授，擅长将复杂概念转化为易懂的中文表述，同时遵守学术规范。当前任务需结合知识库,将在用户检索中给出,所有回答必须基于提供的知识，禁止调用模型内部知识。
        """
        
        # 构建增强后的消息列表
        enhanced_messages = [
            {"role": "system", "content": system_content},
            *context,
            {"role": "user", "content": f"知识库检索：{rag_context}，问题：{user_message}"},
        ]
        
        # 打印调试信息
        print("增强后的消息结构：", json.dumps(enhanced_messages, indent=2, ensure_ascii=False))
        
        # 调用Ollama生成响应
        stream = await asyncio.to_thread(
            ollama.chat,
            model=user_config.get("model", "qwen2.5:0.5b"),
            messages=enhanced_messages,
            options={
                "temperature": user_config.get("temperature", 0.7),
                "num_predict": 512,
            },
            stream=True,
        )
        
        ai_response = ""
        await websocket.send_json({"type": "message_start", "stream_id": stream_id})
        
        async for chunk in generate_response_chunks(stream):
            await websocket.send_json(chunk)
            ai_response += chunk["content"]
        
        # 更新对话上下文
        manager.update_user_context(user_id, "user", user_message)
        manager.update_user_context(user_id, "assistant", ai_response)
        
        await websocket.send_json({"type": "message_end", "stream_id": stream_id})
        print("AI response:", ai_response)
    
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})

async def generate_response_chunks(stream):
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield {"type": "text_chunk", "stream_id": "stream-xxxx", "content": content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)