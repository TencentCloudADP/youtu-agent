import React, { type FC, type KeyboardEvent, useRef, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

export type ChatInputLoadingState = "loading" | "ready" | "hide";

interface ChatInputProps {
  inputValue: string;
  onInputChange: (value: string) => void;
  onSendMessage: (message: string) => void;
  isModelResponding: boolean;
  inputRef: React.RefObject<HTMLInputElement | null>;
  currentConfig?: string;
  onConfigSelect?: (config: string) => void;
  handleAddConfig?: () => void;
  getConfigList: () => void;
  chatInputLoadingState: ChatInputLoadingState;
  setChatInputLoadingState: (state: ChatInputLoadingState) => void;
  availableConfigs: string[];
  uploadEndpoint?: string;
}

type FileStatus = 'uploading' | 'uploaded';
interface SelectedFileItem {
  id: string;
  name: string;
  file: File;
  status: FileStatus;
  hovering: boolean;
  serverFilename?: string;
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
  uploadEndpoint,
}) => {
  const { t } = useTranslation();
  const sendWithRelated = () => {
    // Collect uploaded files' server filenames
    const uploaded = selectedFiles.filter((f) => f.status === 'uploaded' && !!f.serverFilename);
    if (uploaded.length === 0) {
      onSendMessage(inputValue);
      return;
    }
    const names = uploaded.map((f) => f.serverFilename).join(', ');
    const prefix = `${t('app.relatedFiles', 'Related Files')}: ${names}`;
    const composed = inputValue ? `${prefix}\n\n${inputValue}` : prefix;
    onSendMessage(composed);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !isModelResponding) {
      e.preventDefault();
      sendWithRelated();
    }
  };

  const configRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFiles, setSelectedFiles] = useState<SelectedFileItem[]>([]);

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

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  const uploadFile = async (item: SelectedFileItem) => {
    try {
      const form = new FormData();
      form.append('file', item.file);
      const uploadUrl = uploadEndpoint ?? `${window.location.origin}/upload`;
      const res = await fetch(uploadUrl, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) {
        throw new Error(`Upload failed: ${res.status}`);
      }
      const data = await res.json();
      const serverFilename = data?.filename as string | undefined;
      setSelectedFiles((prev) =>
        prev.map((it) =>
          it.id === item.id ? { ...it, status: 'uploaded', serverFilename } : it
        )
      );
    } catch (err) {
      console.error('Upload error:', err);
      // Remove the failed item from the list
      setSelectedFiles((prev) => prev.filter((it) => it.id !== item.id));
    }
  };

  const handleFileChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const newItems: SelectedFileItem[] = Array.from(files).map((f, idx) => ({
      id: `${Date.now()}_${idx}_${f.name}`,
      name: f.name,
      file: f,
      status: 'uploading',
      hovering: false,
    }));

    setSelectedFiles((prev) => [...prev, ...newItems]);

    // Start real uploads
    newItems.forEach((item) => {
      void uploadFile(item);
    });

    // Reset input so selecting the same file again still triggers change
    e.currentTarget.value = '';
  };

  const handleHoverChange = (id: string, hovering: boolean) => {
    setSelectedFiles((prev) => prev.map((it) => (it.id === id ? { ...it, hovering } : it)));
  };

  const handleRemoveFile = (id: string) => {
    setSelectedFiles((prev) => prev.filter((it) => it.id !== id));
  };

  const FileStatusComponent: FC<{
    item: SelectedFileItem;
    onHover: (id: string, hovering: boolean) => void;
    onRemove: (id: string) => void;
  }> = ({ item, onHover, onRemove }) => (
    <div className="file-chip">
      <span
        className={`file-chip-icon ${item.hovering ? 'is-hovering' : ''}`}
        onMouseEnter={() => onHover(item.id, true)}
        onMouseLeave={() => onHover(item.id, false)}
        onClick={() => item.hovering && onRemove(item.id)}
        title={item.hovering ? t('app.delete', '删除') : ''}
      >
        {item.hovering ? (
          <i className="fas fa-times"></i>
        ) : item.status === 'uploading' ? (
          <i className="breathing-circle fa fa-circle"></i>
        ) : (
          <i className="fas fa-file"></i>
        )}
      </span>
      <span className="file-chip-name">{item.name}</span>
    </div>
  );

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
      {/* File status area */}
      {selectedFiles.length > 0 && (
        <div className="file-area">
          {selectedFiles.map((item) => (
            <FileStatusComponent
              key={item.id}
              item={item}
              onHover={handleHoverChange}
              onRemove={handleRemoveFile}
            />
          ))}
        </div>
      )}

      <div className="chat-input-wrapper">
        <div className="input-group">
          <button
            className="send-button attach-button"
            onClick={handleAttachClick}
            disabled={isModelResponding}
            title={isModelResponding ? t('app.aiThinking') : t('app.attachFile', 'Attach file')}
          >
            <i className="fas fa-paperclip"></i>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden-file-input"
            multiple
            onChange={handleFileChange}
          />
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
            onClick={sendWithRelated}
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
