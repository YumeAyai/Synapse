import { useState, useEffect, useRef } from 'react';
import { useWebSocketContext } from './WebSocketContext';

export default function AdminPanel() {
  const [temperature, setTemperature] = useState(0.7);
  const { send, isConnected, reconnecting, status } = useWebSocketContext();

  // 温度控制带防抖
  useEffect(() => {
    let timer: number | undefined;
    if (isConnected) {
      timer = setTimeout(() => {
        send({
          type: 'update_config',
          payload: {
            temperature,
            model: 'qwen2.5:0.5b',
          },
        });
      }, 500);
    }
    return () => clearTimeout(timer);
  }, [temperature, send, isConnected]);

  // 10s 轮询
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      if (!isConnected) return;

      // 发送获取状态请求
      send({
        type: 'get_status',
        stream_id: `user-${Date.now()}`,
        session_id: localStorage.getItem('session_id'), // 从本地存储获取 session_id
      });
    }, 10000);

    return () => clearInterval(interval);
  }, [send, isConnected]);

  return (
    <div className="admin-panel">
      <h2>系统控制台（实时）</h2>
      <div className="status-card">
        <p>模型状态: {status.model_status || (reconnecting ? '正在连接...' : '已连接')}</p>
        <p>当前温度: {status.temperature}</p>
        <p>累计请求: {status.requests_count}</p>
      </div>

      <div className="control-group">
        <label>
          生成温度：
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
          />
          {temperature}
        </label>
      </div>
    </div>
  );
}