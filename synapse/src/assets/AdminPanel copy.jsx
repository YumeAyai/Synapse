import React, { useEffect, useState } from 'react';
import axios from 'axios';

function AdminPanel() {
  const [status, setStatus] = useState({});
  const [temperature, setTemperature] = useState(0.7);

  // 状态探针核心逻辑（保持原样）
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await axios.get('/api/status');
        setStatus(res.data);
      } catch (err) {
        console.error('状态获取失败:', err);
      }
    };

    fetchStatus(); // 立即获取
    const interval = setInterval(fetchStatus, 5000); // 每5秒轮询
    return () => clearInterval(interval);
  }, []); // 空依赖数组表示只运行一次

  // 新增的温度控制逻辑（独立存在）
  useEffect(() => {
    const updateTemperature = async () => {
      try {
        await axios.put('/api/config', {
          temperature,
          model: 'deepseek-r1:7b'
        });
      } catch (err) {
        console.error('温度更新失败:', err);
      }
    };

    const timer = setTimeout(updateTemperature, 500);
    return () => clearTimeout(timer);
  }, [temperature]); // 温度变化时触发

  return (
    <div className="admin-panel">
      <h2>系统控制台</h2>
      <div className="status-card">
        <p>模型状态: {status.model_status}</p>
        <p>累计请求: {status.requests_count}</p>
        <p>最后错误: {status.last_error || '无'}</p>
      </div>

      <div className="control-group">
        <label>
          生成温度：
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(e.target.value)}
          />
          {temperature}
        </label>
      </div>
    </div>
  );
}

export default AdminPanel;