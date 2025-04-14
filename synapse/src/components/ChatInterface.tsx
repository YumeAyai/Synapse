import { useState, useEffect, useRef } from 'react';
import { useWebSocketContext } from './WebSocketContext';
import DOMPurify from 'dompurify';
import { marked } from 'marked';  // 用于渲染 Markdown

// const MESSAGE_TYPES = {
//   USER: 'user',
//   ASSISTANT: 'assistant',
//   SYSTEM: 'system',
//   THINKING: 'thinking',
//   CODE: 'code',
//   LOADING: 'loading'
// };

export default function ChatInterface() {
  const [input, setInput] = useState(''); // 用户输入的消息
  const messagesEndRef = useRef<HTMLDivElement>(null); // 用于滚动到底部
  const { send, messages } = useWebSocketContext(); // 从上下文中获取 WebSocket 的消息和发送函数

  const scrollToBottom = () => {
    // 每次新消息来时，滚动到底部
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom(); // 新消息到来时自动滚动
  }, [messages]); // 每当 messages 改变时触发

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return; // 如果输入为空，则不发送
    setInput(''); // 清空输入框

    // 发送用户消息
    send({
      type: 'user_message',
      content: input,
      stream_id: `user-${Date.now()}`,
      session_id: localStorage.getItem('session_id'), // 从本地存储获取 session_id
    });

    // 将用户消息添加到消息流中
    // 在这里直接将用户消息添加到对话流
    // 你也可以在收到响应后将其追加（由 WebSocket 响应）
    // 注意：此时流式加载助手消息
  };

  // 定义对话结构
  const renderMessageContent = (message: { user?: { content: string }; assistant?: { content: string }; type: string }) => {
    let content = '';
    if (message.user) content = message.user.content;
    if (message.assistant) content = message.assistant.content;

    const htmlContent = marked(content);

    // 使用 DOMPurify 防止 XSS 攻击

    // <think> 标签用于表示助手正在思考,转换成think css类
    // content = content.replace(/<think>/g, '<span class="think">');

    // <code> 标签用于表示代码块
    // content = content.replace(/<code>/g, '<pre><code>');

    // 清洁内容以防 XSS 攻击
    const sanitized = DOMPurify.sanitize(htmlContent as string, {
      ALLOWED_TAGS: ['span', 'pre', 'code', 'b', 'i', 'em', 'strong', 'think'], // 允许这些标签
      ALLOWED_ATTR: ['class'] // 允许 class 属性
    });

    if (message.user) {
      return <div className="user-message" dangerouslySetInnerHTML={{ __html: sanitized }} />;
    }

    if (message.assistant) {
      return <div className="assistant-message" dangerouslySetInnerHTML={{ __html: sanitized }} />;
    }

    return null; // 默认为 null，不渲染未知类型
  };

  return (
    <div className="chat-container">
      {messages.length > 0 && (
        <div className="message-list">
          {messages.map((message, index) => (
            <div key={index} className="message-item">
              {renderMessageContent(message)} {/* 根据消息类型渲染内容 */}
            </div>
          ))}
          <div ref={messagesEndRef} /> {/* 保证滚动到底部 */}
        </div>
      )}

      <form className="input-area" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)} // 更新输入框内容
          placeholder="输入消息..."
          autoFocus
        />
        <button type="submit">发送</button>
      </form>
    </div>
  );
}
