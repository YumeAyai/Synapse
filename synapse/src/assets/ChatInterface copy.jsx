// src/components/ChatInterface.jsx
import React, { useState } from 'react';
import axios from 'axios';
import DOMPurify from 'dompurify'; // 安装：npm install dompurify

export default function ChatInterface() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState([]);
  const [loading, setLoading] = useState(false);

  const parseResponse = (text) => {
    const result = [];
    // 用正则表达式解析特殊格式
    const thinkMatch = text.match(/(.*?)<think>(.*?)<\/think>(.*)/s);
    if (thinkMatch) {
      result.push({ type: 'text', content: thinkMatch[1] });
      result.push({ type: 'think', content: thinkMatch[2] });
      result.push({ type: 'text', content: thinkMatch[3] });
    } else if (text.includes('```')) {
      result.push({ type: 'code', content: text });
    } else {
      result.push({ type: 'text', content: text });
    }
    return result;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAnswer([]);
    
    try {
      const response = await fetch('http://localhost:8000/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question,
          stream: true // 启用流式传输
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value);
        const parts = buffer.split('\n');
        buffer = parts.pop();

        parts.forEach(part => {
          if (part.startsWith('data:')) {
            const content = JSON.parse(part.replace('data:', '')).answer;
            setAnswer(prev => [...prev, ...parseResponse(content)]);
          }
        });
      }
    } catch (err) {
      setAnswer([{ type: 'error', content: err.message }]);
    }
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <form onSubmit={handleSubmit}>
        {/* 保持原有输入表单 */}
      </form>
      <div className="answer-box">
        {answer.map((item, index) => (
          <div key={index} className={`response-${item.type}`}>
            {item.type === 'code' ? (
              <pre dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(item.content) }} />
            ) : (
              <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(item.content) }} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}