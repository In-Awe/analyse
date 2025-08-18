
import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, MessageSender } from '../types';
import { SendIcon, GeminiIcon, UserIcon, LoadingSpinner } from './icons';

interface ChatPanelProps {
  history: ChatMessage[];
  onSendMessage: (message: string) => void;
  isLoading: boolean;
}

const ChatPanel: React.FC<ChatPanelProps> = ({ history, onSendMessage, isLoading }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSendMessage(input);
      setInput('');
    }
  };

  return (
    <div className="flex flex-col h-full p-4">
      <h2 className="text-xl font-semibold text-cyan-400 mb-2 border-b border-gray-700 pb-2">AI Analyst Chat</h2>
      <div className="flex-grow overflow-y-auto pr-2">
        <div className="space-y-4">
          {history.map((msg, index) => (
            <div key={index} className={`flex items-start gap-3 ${msg.sender === MessageSender.USER ? 'justify-end' : ''}`}>
              {msg.sender === MessageSender.GEMINI && <GeminiIcon />}
              <div className={`max-w-md rounded-lg px-4 py-2 ${msg.sender === MessageSender.USER ? 'bg-cyan-600 text-white' : 'bg-gray-700'}`}>
                <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
              </div>
              {msg.sender === MessageSender.USER && <UserIcon />}
            </div>
          ))}
          {isLoading && (
            <div className="flex items-start gap-3">
                <GeminiIcon />
                <div className="max-w-md rounded-lg px-4 py-2 bg-gray-700 flex items-center">
                    <LoadingSpinner />
                    <span className="ml-2 text-sm">Analyzing...</span>
                </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      <form onSubmit={handleSubmit} className="mt-4 flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the chart patterns..."
          className="flex-grow bg-gray-700 border border-gray-600 rounded-full py-2 px-4 focus:ring-2 focus:ring-cyan-500 focus:outline-none"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-full p-2 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-400"
        >
          <SendIcon />
        </button>
      </form>
    </div>
  );
};

export default ChatPanel;
