import React, { createContext, useState, useEffect, useContext, useCallback } from 'react';

interface Message {
  user?: { content: string };  // 用户消息
  assistant?: { content: string };  // 助手消息
  type: 'user' | 'assistant' | 'system' | 'thinking' | 'code';
  timestamp: number;
  complete?: boolean;  // 是否完成消息
}

interface SystemStatus {
  model_status?: string;
  temperature?: number;
  requests_count?: number;
}

interface WebSocketContextType {
  send: (message: any) => void;
  messages: Message[]; // 存储消息
  isConnected: boolean;
  reconnecting: boolean;
  status: SystemStatus;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocketContext = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useWebSocketContext must be used within a WebSocketProvider");
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
  url: string;
  reconnectDelay?: number;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children, url, reconnectDelay = 5000 }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<SystemStatus>({});  // 系统状态

  // 创建 WebSocket 连接
  const createWebSocket = useCallback(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setIsConnected(true);
      setReconnecting(false);
    };

    ws.onclose = () => {
      setIsConnected(false);
      if (!reconnecting) {
        setReconnecting(true);
        setTimeout(createWebSocket, reconnectDelay); // 重连延迟
      }
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // 处理系统状态更新
      if (data.type === 'system_status') {
        const payload = data.payload as SystemStatus;
        setStatus(payload);
      }

      // 处理流式消息
      if (data.type === 'message_start') {
        setMessages(prev => [
          ...prev,
          {
            assistant: { content: '' },
            type: 'assistant',
            timestamp: Date.now(),
            complete: false // 标记消息不完整
          }
        ]);
      }

      if (data.type === 'text_chunk') {
        setMessages(prev => prev.map(msg =>
          msg.type === 'assistant' && !msg.complete
            ? { ...msg, assistant: { content: msg.assistant?.content + data.content } }
            : msg
        ));
      }

      if (data.type === 'message_end') {
        setMessages(prev => prev.map(msg =>
          msg.type === 'assistant'
            ? { ...msg, complete: true }
            : msg
        ));
      }
    };

    setSocket(ws);
  }, [url, reconnectDelay, reconnecting]);

  // 组件挂载时建立连接
  useEffect(() => {
    if (!socket) {
      createWebSocket();  // 建立 WebSocket 连接
    }

    return () => {
      if (socket) {
        socket.close();  // 清理 WebSocket 连接
      }
    };
  }, [createWebSocket, socket]);

  // 监听 messages 更新并打印
  useEffect(() => {
    console.log("Updated messages array:", messages);
  }, [messages]); // 每次 messages 数组变化时打印

  const send = useCallback((message: any) => {
    if (socket && isConnected) {
      if (message.type === 'user_message') {
        setMessages(prev => [
          ...prev,
          {
            user: { content: message.content },
            type: 'user',
            timestamp: message.timestamp
          }
        ]);
      }
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket未连接，无法发送消息');
    }
  }, [socket, isConnected]);

  return (
    <WebSocketContext.Provider value={{ send, messages, isConnected, reconnecting, status }}>
      {children}
    </WebSocketContext.Provider>
  );
};
