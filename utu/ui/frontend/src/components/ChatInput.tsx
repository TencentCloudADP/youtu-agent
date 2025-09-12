import React, { type FC, type KeyboardEvent, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

export type ChatInputLoadingState = "loading" | "ready" | "hide";

interface ChatInputProps {
  inputValue: string;
  onInputChange: (value: string) => void;
  onSendMessage: () => void;
  isModelResponding: boolean;
  inputRef: React.RefObject<HTMLInputElement | null>;
  currentConfig?: string;
  onConfigSelect?: (config: string) => void;
  handleAddConfig?: () => void;
  getConfigList: () => void;
  chatInputLoadingState: ChatInputLoadingState;
  setChatInputLoadingState: (state: ChatInputLoadingState) => void;
  availableConfigs: string[];
}

// Mock config options - replace with actual config fetching logic if needed
// const CONFIG_OPTIONS = [
//   'Default Config',
//   'Research Mode',
//   'Coding Mode',
//   'Debug Mode',
//   'Production Mode'
// ];

const ChatInput: FC<ChatInputProps> = ({
  inputValue,
  onInputChange,
  onSendMessage,
  isModelResponding,
  inputRef,
  currentConfig = '',
  onConfigSelect,
  handleAddConfig,
  getConfigList,
  chatInputLoadingState,
  setChatInputLoadingState,
  availableConfigs,
}) => {
  const { t } = useTranslation();
  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !isModelResponding) {
      e.preventDefault();
      onSendMessage();
    }
  };

  const configRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (configRef.current && !configRef.current.contains(event.target as Node)) {
        // setIsConfigOpen(false);
        setChatInputLoadingState("hide");
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleConfigClick = () => {
    getConfigList();
    // setChatInputLoadingState("loading");
  };

  const handleConfigSelect = (config: string) => {
    onConfigSelect?.(config);
    setChatInputLoadingState("hide");
  };

  return (
    <div className="chat-input-container">
      {/* Config section is now hidden by default */}
      {false && currentConfig && (
        <div className="config-display" ref={configRef}>
          <div className="config-header">
            <span className="config-label">Config: </span>
            <span 
              className="config-value" 
              onClick={handleConfigClick}
            >
              {currentConfig}
            </span>
          </div>
          {chatInputLoadingState === "loading" && (
            <div className="config-options-container">
              <div className="config-option config-option-loading">
                Loading...
              </div>
            </div>
          )}
          {chatInputLoadingState === "ready" && (
            <div className="config-options-container">
              {availableConfigs.map((config) => (
                <div 
                  key={config}
                  className={`config-option ${config === currentConfig ? 'active' : ''}`}
                  onClick={() => handleConfigSelect(config)}
                >
                  {config}
                </div>
              ))}
              <div className="config-option config-option-add" onClick={handleAddConfig}>
                +
              </div>
            </div>
          )}
        </div>
      )}
      <div className="chat-input-wrapper">
        <div className="input-group">
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isModelResponding ? t('app.aiThinking') : t('app.inputPlaceholder')}
            disabled={isModelResponding}
          />
          <button
            className="send-button"
            onClick={onSendMessage}
            disabled={isModelResponding}
            title={isModelResponding ? t('app.aiThinking') : t('app.sendMessage')}
          >
            {isModelResponding ? (
              <i className="breathing-circle fa fa-circle"></i>
            ) : (
              <i className="fas fa-paper-plane"></i>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
