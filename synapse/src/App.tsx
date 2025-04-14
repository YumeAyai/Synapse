// src/App.jsx
import { WebSocketProvider } from './components/WebSocketContext';
import AdminPanel from './components/AdminPanel';
import ChatInterface from './components/ChatInterface';
import './App.css';

const App = () => {
  return (
    <WebSocketProvider url="ws://localhost:8000/ws/chat">
      <div>
        <h1>Synapse</h1>
        <div className="App">
          <ChatInterface />
          <AdminPanel />
        </div>
      </div>
    </WebSocketProvider>
  );
};

export default App;
