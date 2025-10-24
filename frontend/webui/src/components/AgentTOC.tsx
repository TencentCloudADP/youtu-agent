import React from 'react';
import type { Message } from '../types/message';

interface AgentTOCProps {
  messages: Message[];
  onNavigate: (id: number) => void;
}

const AgentTOC: React.FC<AgentTOCProps> = ({ messages, onNavigate }) => {
  // Filter only agent messages
  const agentMessages = messages.filter(
    (msg) => msg.type === 'new_agent' && typeof msg.content === 'string'
  );

  if (agentMessages.length === 0) {
    return null;
  }

  return (
    <div className="agent-toc">
      <div className="agent-toc-header">Agents</div>
      <div className="agent-toc-list">
        {agentMessages.map((msg) => (
          <div 
            key={msg.id} 
            className="agent-toc-item"
            onClick={() => onNavigate(msg.id)}
          >
            <i className="fas fa-robot agent-toc-icon"></i>
            <span className="agent-toc-name">{msg.content as string}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentTOC;
