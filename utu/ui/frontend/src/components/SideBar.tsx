import React, { useEffect, useRef, useState } from 'react';
import './SideBar.css';
import type { Message } from '../types/message';
import { useTranslation } from 'react-i18next';

interface SideBarProps {
  isOpen: boolean;
  onClose: () => void;
  messages?: Message[];
  onNavigate?: (id: number) => void;
  currentConfig?: string;
  agentType?: 'simple' | 'orchestra' | 'other';
  subAgents?: string[] | null;
  onConfigSelect?: (config: string) => void;
  handleAddConfig?: () => void;
  getConfigList: () => void;
  availableConfigs: string[];
  showNewChatButton?: boolean;
  showAgentConfigs?: boolean;
}

// Helper function to convert path to a more readable format
const getFilename = (path: string) => {
  // Remove file extension and path
  const filename = path.split('/').pop() || '';
  const nameWithoutExt = filename.replace(/\.ya?ml$/, '');
  
  // Convert snake_case or kebab-case to Title Case
  return nameWithoutExt
    .split(/[_-]/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join('');
};

// Custom tooltip component
const Tooltip = ({ content, children }: { content: string; children: React.ReactNode }) => {
  const [isHovered, setIsHovered] = React.useState(false);
  const [position, setPosition] = React.useState({ top: 0, left: 0 });
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (ref.current && isHovered) {
      const rect = ref.current.getBoundingClientRect();
      setPosition({
        left: rect.right + window.scrollX + 8, // 8px offset from the right edge
        top: rect.top + window.scrollY + (rect.height / 2) // Vertical center
      });
    }
  }, [isHovered]);
  
  return (
    <div 
      ref={ref}
      className="tooltip-container"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {children}
      {isHovered && (
        <div 
          className="tooltip"
          style={{
            left: position.left,
            top: position.top,
          }}
        >
          {content}
        </div>
      )}
    </div>
  );
};

const SideBar: React.FC<SideBarProps> = ({
  isOpen,
  onClose,
  messages = [],
  onNavigate = () => {},
  currentConfig = '',
  agentType = 'simple',
  subAgents = null,
  onConfigSelect = () => {},
  handleAddConfig = () => {},
  getConfigList,
  availableConfigs = [],
  showNewChatButton = true,
  showAgentConfigs = true
}) => {
  const sidebarRef = useRef<HTMLDivElement>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const { t } = useTranslation();
  const filteredConfigs = availableConfigs
    .filter(config => config.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) => getFilename(a).localeCompare(getFilename(b)));

  // Load configs only once when component mounts
  useEffect(() => {
    if (availableConfigs.length === 0) {
      getConfigList();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty dependency array means this runs once on mount

  // Close sidebar when clicking outside on mobile
  useEffect(() => {
    if (window.innerWidth > 768) return; // Only for mobile
    
    const handleClickOutside = (event: MouseEvent) => {
      if (sidebarRef.current && !sidebarRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);

  const handleNewChat = () => {
    window.open(window.location.href, '_blank');
  };

  return (
    <div 
      ref={sidebarRef}
      className={`sidebar ${isOpen ? 'sidebar-open' : ''}`}
    >
      <div className="sidebar-content">
        {showNewChatButton && (
          <>
            <div className="sidebar-section">
              <button 
                className="sidebar-button primary"
                onClick={handleNewChat}
              >
                <i className="fas fa-plus" />
                {t('sidebar.newChat')}
              </button>
            </div>
            
            <div className="sidebar-divider" />
          </>
        )}
        
        {/* Agent History Section */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">{t('sidebar.agentHistoryTitle')}</div>
          <div className="agent-toc-list">
            {messages.filter(msg => msg.type === 'new_agent' && typeof msg.content === 'string').length > 0 ? (
              messages
                .filter((msg) => msg.type === 'new_agent' && typeof msg.content === 'string')
                .map((msg) => (
                  <div 
                    key={msg.id}
                    className="sidebar-button sidebar-button-text agent-toc-item"
                    onClick={() => onNavigate(Number(msg.id))}
                  >
                    <i className="fas fa-robot" style={{ marginRight: '8px' }}></i>
                    {msg.content as string}
                  </div>
                ))
            ) : (
              <div className="sidebar-button-text" style={{ padding: '8px 16px', color: 'var(--color-subtle-text, #6c757d)' }}>
                {t('sidebar.noHistoryShort')}
              </div>
            )}
          </div>
        </div>

        <div className="sidebar-divider" />
        
        {/* Current Config Section */}
        {currentConfig ? (
          <div className="sidebar-section">
            <div className="sidebar-section-title">{t('sidebar.currentConfigTitle')}</div>
            <div className="current-config-display">
              <div className="current-config-header">
                <i className="fas fa-check-circle"></i>
                <span className="current-config-title">{getFilename(currentConfig)}</span>
              </div>
              <div className="current-config-path">
              {currentConfig}
            </div>
            {agentType === 'orchestra' && subAgents && subAgents.length > 0 && (
              <div className="sub-agents-section">
                <div className="sub-agents-title">{t('sidebar.subAgentsTitle')}</div>
                <div className="sub-agents-list">
                  {subAgents.map((agent, index) => (
                    <div key={index} className="sub-agent-item">
                      <span className="sub-agent-name">{agent}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            </div>
            <div 
              className="sidebar-button-text agent-toc-item add-config-button"
              onClick={handleAddConfig}
            >
              <i className="fas fa-plus" style={{ marginRight: '6px' }}></i>
              {t('sidebar.addNewConfig')}
            </div>
          </div>
        ) : null}
        
        {showAgentConfigs && (
          <>
            <div className="sidebar-divider" style={{ margin: '12px 0' }} />

            {/* Available Configs Section */}
            <div className="sidebar-section">
              <div className="sidebar-section-header">
                <div className="sidebar-section-title">{t('sidebar.availableConfigsTitle')}</div>
                <button 
                  className={`refresh-button ${isRefreshing ? 'refreshing' : ''}`}
                  onClick={() => {
                    setIsRefreshing(true);
                    getConfigList();
                    setTimeout(() => setIsRefreshing(false), 1000);
                  }}
                  disabled={isRefreshing}
                  title={t('sidebar.refreshConfigs')}
                >
                  <i className="fas fa-sync-alt" />
                </button>
              </div>
              <div className="search-box-container" style={{ marginBottom: '12px' }}>
                <div className="search-box">
                  <i className="fas fa-search search-icon" />
                  <input
                    type="text"
                    placeholder={t('sidebar.searchConfigs')}
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="search-input"
                  />
                  {searchTerm && (
                    <button 
                      className="clear-search" 
                      onClick={() => setSearchTerm('')}
                      title={t('sidebar.clearSearch')}
                    >
                      <i className="fas fa-times" />
                    </button>
                  )}
                </div>
              </div>
              <div className="config-list">
                {filteredConfigs.length > 0 ? (
                  filteredConfigs.map((config) => (
                    <Tooltip key={config} content={config}>
                      <div
                        className={`config-toc-item ${currentConfig === config ? 'active' : ''}`}
                        onClick={() => onConfigSelect(config)}
                      >
                        <div className="config-list-item">
                          <div className="config-icon-container">
                            {config.includes('generated/') ? (
                              <i className="fas fa-robot config-icon" title={t('sidebar.generatedConfigTooltip')} />
                            ) : config.includes('examples/') ? (
                              <i className="fas fa-flask config-icon" title={t('sidebar.exampleConfigTooltip')} />
                            ) : null}
                          </div>
                          <span className="config-name">
                            {getFilename(config)}
                          </span>
                        </div>
                      </div>
                    </Tooltip>
                  ))
                ) : (
                  <div className="sidebar-button-text" style={{ padding: '8px 16px', color: 'var(--color-subtle-text, #6c757d)' }}>
                    {availableConfigs.length === 0 ? t('sidebar.noConfigsShort') : t('sidebar.noMatchingConfigs')}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default SideBar;
