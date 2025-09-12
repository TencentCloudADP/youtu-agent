import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useChatWebSocket } from './services/websocket.service';
import { ReadyState } from 'react-use-websocket';
import MessageComponent from './components/MessageComponent';
import ChatInput from './components/ChatInput';
import SideBar from './components/SideBar';
import HamburgerButton from './components/HamburgerButton';
import { simulateEvents } from './placeholderData';
import logoUrl from './assets/pic.png';
import logoSquareUrl from './assets/logo-square.png';
import { useTranslation } from 'react-i18next';
import type {
  Event,
  TextDeltaContent,
  ExampleContent,
  PlanItem,
  WorkerItem,
  ReportItem,
  OrchestraContent,
  InitContent,
  SwitchAgentContent,
  ListAgentsContent,
  AskContent,
  GeneratedAgentContent
} from './types/events';
import type {
  UserRequest,
} from './types/requests';
import type {
  Message,
  ToolCallMessage,
} from './types/message';
import type { ChatInputLoadingState } from './components/ChatInput';

// Initial messages will be created inside the component using i18n

// 根据当前协议自动选择 ws:// 或 wss://
const defaultProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const defaultWsUrl = import.meta.env.VITE_WS_URL || `${defaultProtocol}//${window.location.host}/ws`;


const App: React.FC = () => {
  const { t, i18n } = useTranslation();
  const getSystemPrefersDark = () => window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const getInitialTheme = () => (localStorage.getItem('theme') as 'light' | 'dark' | null) || (getSystemPrefersDark() ? 'dark' : 'light');
  const [theme, setTheme] = useState<'light' | 'dark'>(getInitialTheme());
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  // Track mobile view for responsive behavior
  const [messages, setMessages] = useState<Message[]>([{
    id: 1,
    content: t('app.welcome'),
    sender: 'assistant',
    timestamp: new Date()
  }]);
  const [exampleQuery, setExampleQuery] = useState<string[]>([]);
  const [hideExampleQuery, setHideExampleQuery] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [wsUrl, setWsUrl] = useState(localStorage.getItem('wsUrl') || defaultWsUrl);
  const { sendQuery, sendRequest, lastMessage, readyState } = useChatWebSocket(wsUrl);
  const [inputValue, setInputValue] = useState('');
  const [isModelResponding, setIsModelResponding] = useState(false);
  // const messagesEndRef = useRef<HTMLDivElement>(null);
  const mainContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [currentConfig, setCurrentConfig] = useState<string>("");
  const [chatInputLoadingState, setChatInputLoadingState] = useState<ChatInputLoadingState>("hide");
  const [hasUserSentMessage, setHasUserSentMessage] = useState(false);
  const [isUserAtBottom, setIsUserAtBottom] = useState(true);
  const [availableConfigs, setAvailableConfigs] = useState<string[]>([]);
  const [_isGeneratingAgent, setIsGeneratingAgent] = useState(false);
  const [askId, setAskId] = useState<string | null>(null);
  const [agentType, setAgentType] = useState<'simple' | 'orchestra' | 'other'>('simple');
  const [subAgents, setSubAgents] = useState<string[] | null>(null);

  const [showNewChatButton, setShowNewChatButton] = useState(false);
  const [showAgentConfigs, setShowAgentConfigs] = useState(false);

  const getConfigList = () => {
    let request: UserRequest = {
      type: 'list_agents',
      content: null,
    }
    setChatInputLoadingState("loading");
    sendRequest(request);
  }

  const handleEvent = (event: Event) => {
    console.log('current messages', messages);
    
    if (event.type === 'example') {
      const data = event.data as ExampleContent;
      const query = data.query;
      if (query) {
        setExampleQuery([query]);
      }
      return;
    }

    // handle raw event
    if (event.type === 'raw') {
      const data = event.data as TextDeltaContent;

      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];

        if (data.type == "tool_call") {
          // create new message
          const newMessage: Message = {
            id: Date.now(),
            content: {
              toolName: data.delta || "",
              toolCallArgument: data.argument || "",
              toolCallOutput: "",
              callid: data.callid || "",
            },
            sender: 'assistant',
            timestamp: new Date(),
            type: "tool_call",
            inprogress: true,
            requireConfirm: event.requireConfirm,
          };
          return [...prev, newMessage];
        }

        if (data.type == "tool_call_output") {
          // find tool_call with same callid
          const tool_call_message = prev.find((message) => message.type == "tool_call" && (message.content as ToolCallMessage).callid == data.callid);
          if (tool_call_message) {
            const updatedMessage: Message = {
              ...tool_call_message,
              content: {
                ...(tool_call_message.content as ToolCallMessage),
                toolCallOutput: data.delta,
              },
              inprogress: false,
              requireConfirm: event.requireConfirm,
            };
            // update tool_call_message
            const messageIndex = prev.findIndex(message =>
              message.type === "tool_call" &&
              (message.content as ToolCallMessage).callid === data.callid
            );
            if (messageIndex !== -1) {
              const updatedMessages = [...prev];
              updatedMessages[messageIndex] = updatedMessage;
              return updatedMessages;
            }
            return prev;
          }
        }

        // If last message is in progress and has the same type, update it
        if (lastMessage?.inprogress && (lastMessage.type == data.type)) {
          const updatedMessage: Message = {
            ...lastMessage,
            content: lastMessage.content + data.delta,
            inprogress: data.inprogress,
            requireConfirm: event.requireConfirm,
          };
          return [...prev.slice(0, -1), updatedMessage];
        }

        if (data.type == "text" || data.type == "reason") {
          // Otherwise, add a new non-tool message
          const newMessage: Message = {
            id: Date.now(),
            content: data.delta,
            sender: 'assistant',
            timestamp: new Date(),
            type: data.type,
            inprogress: data.inprogress,
            requireConfirm: event.requireConfirm,
          };
          return [...prev, newMessage];
        }

        return prev
      });
    } 
    // handle new agent event
    else if (event.type === 'new') {
      const data = event.data as { name: string };
      const message: Message = {
        id: Date.now() + 1,
        content: data.name,
        sender: 'assistant',
        timestamp: new Date(),
        type: 'new_agent',
        inprogress: true,
        requireConfirm: event.requireConfirm,
      };
      setMessages(prev => [...prev, message]);
      
      // Simulate agent initialization
      setTimeout(() => {
        setMessages(prev => {
          const updatedMessages = [...prev];
          const agentMessageIndex = updatedMessages.findIndex(m => m.id === message.id);
          if (agentMessageIndex !== -1) {
            updatedMessages[agentMessageIndex] = {
              ...updatedMessages[agentMessageIndex],
              inprogress: false,
            };
          }
          return updatedMessages;
        });
      }, 1500);
    } 
    // handle orchestra event
    else if (event.type === 'orchestra') {
      const data = event.data as OrchestraContent;
      if (data.type === 'plan') {
        const item = data.item as PlanItem;
        // setPinnedPlan(item);
        const message: Message = {
          id: Date.now() + 1,
          content: item,
          sender: 'assistant',
          timestamp: new Date(),
          type: 'plan',
          requireConfirm: event.requireConfirm,
        }
        setMessages(prev => [...prev, message]);
      } else if (data.type === 'worker') {
        const item = data.item as WorkerItem;
        const message: Message = {
          id: Date.now() + 1,
          content: "task: \n" + item.task + '\n\n' + "output: \n" + item.output,
          sender: 'assistant',
          timestamp: new Date(),
          type: 'worker',
          requireConfirm: event.requireConfirm,
        };
        setMessages(prev => [...prev, message]);
      } else if (data.type === 'report') {
        const item = data.item as ReportItem;
        const message: Message = {
          id: Date.now() + 1,
          content: "report: \n" + item.output,
          sender: 'assistant',
          timestamp: new Date(),
          type: 'report',
          requireConfirm: event.requireConfirm,
        };
        setMessages(prev => [...prev, message]);
      }
    } 
    // handle finish event
    else if (event.type === 'finish') {
      setIsModelResponding(false); // Call save function on finish
    } 
    // handle generated config event
    else if (event.type === 'generated_agent_config') {
      let data = event.data as GeneratedAgentContent;
      let config = data.config_content;
      let filename = data.filename;
      const content = "```\n" + config + "\n```\n\n" + "filename: " + '`' + filename + '`';
      const message: Message = {
        id: Date.now() + 1,
        content: content,
        sender: 'assistant',
        timestamp: new Date(),
        type: 'text',
        requireConfirm: event.requireConfirm,
      }
      setMessages(prev => [...prev, message]);
    }
    // handle init event
    else if (event.type === 'init') {
      const initData = event.data as InitContent;
      setCurrentConfig(initData.default_agent);
      setAgentType(initData.agent_type);
      setSubAgents(initData.sub_agents);
      setShowNewChatButton(true);
      setShowAgentConfigs(true);
      
      // Send list_agents command on init
      sendRequest({
        type: 'list_agents',
        content: null
      });
    } 
    // handle switch agent event
    else if (event.type === 'switch_agent') {
      let data = event.data as SwitchAgentContent;
      if (data.ok) {
        setCurrentConfig(data.name);
        setAgentType(data.agent_type);
        setSubAgents(data.sub_agents);
        setIsGeneratingAgent(false);
        // clear messages
        setMessages([{
          id: Date.now(),
          content: t('app.welcome'),
          sender: 'assistant',
          timestamp: new Date()
        }]);
      } else {
        console.error('Switch agent failed');
      }
    } 
    // handle list agents event
    else if (event.type === 'list_agents') {
      let data = event.data as ListAgentsContent;
      setAvailableConfigs(data.agents);
      // Set loading to false when we receive the agents list
      if (chatInputLoadingState === 'loading') {
        setChatInputLoadingState('ready');
      }
    } 
    // handle ask event
    else if (event.type === 'ask') {
      let data = event.data as AskContent;
      let ask_id = data.ask_id;
      setAskId(ask_id);
      const Message: Message = {
        id: Date.now(),
        content: data.question,
        sender: 'assistant',
        timestamp: new Date(),
        type: 'text',
        inprogress: false,
        requireConfirm: false,
      };
      setMessages(prev => [...prev, Message]);
      setIsModelResponding(false);
    }
    // handle gen agent event
    else if (event.type === 'gen_agent') {
      // clean up message
      setMessages([]);
      setIsModelResponding(true);
      // Set agent type to orchestra and specify sub-agents
      setAgentType('orchestra');
      setSubAgents([
        'clarification_agent',
        'tool_selection_agent',
        'instructions_generation_agent',
        'name_generation_agent'
      ]);
      // add a prompt message
      const message: Message = {
        id: Date.now(),
        content: t('app.agentGenerationPrompt'),
        sender: 'assistant',
        timestamp: new Date(),
        type: 'text',
        inprogress: false,
        requireConfirm: false,
      };
      // after 1s
      setTimeout(() => {
        setMessages(prev => [...prev, message]);
        setIsModelResponding(false);
      }, 500);
      setIsGeneratingAgent(true);
      setCurrentConfig("generate_agent");
      // collapse config panel
      setChatInputLoadingState("hide");
    } else {
      console.error('Unknown event type:', event.type);
    }
  }

  useEffect(() => {
    // Focus input on mount
    inputRef.current?.focus();
  }, []);

  // Apply theme class to document root
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Set favicon using logoSquareUrl on mount
  useEffect(() => {
    try {
      let link = document.querySelector("link[rel~='icon']") as HTMLLinkElement | null;
      if (!link) {
        link = document.createElement('link');
        link.rel = 'icon';
        document.head.appendChild(link);
      }
      link.type = 'image/png';
      link.href = logoSquareUrl;
    } catch (e) {
      console.error('Failed to set favicon', e);
    }
  }, []);

  useEffect(() => {
    if (lastMessage !== null) {
      const message_raw = lastMessage;
      const message_json = JSON.parse(message_raw.data);

      // parse to event
      const event: Event = message_json;
      console.log(event);
      handleEvent(event);
      if (isUserAtBottom) {
        scrollToBottom();
      }
    }
  }, [lastMessage]);

  const handleSendMessage = () => {
    if (!inputValue.trim() || isModelResponding) return;

    // Mark that user has sent their first message
    if (!hasUserSentMessage) {
      setHasUserSentMessage(true);
    }

    if (!hideExampleQuery) {
      setHideExampleQuery(true);
    }

    // Add user message to chat
    const userMessage: Message = {
      id: Date.now(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsModelResponding(true);

    if (readyState !== ReadyState.OPEN) {
      console.error("WebSocket is not connected");
      const message: Message = {
        id: Date.now() + 1,
        content: t('app.notConnected'),
        sender: 'assistant',
        timestamp: new Date(),
        type: 'error',
      };
      setMessages(prev => [...prev, message]);

      // simulate event
      simulateEvents(handleEvent);
        
      return;
    }

    if (askId) {
      sendRequest({
        type: 'answer',
        content: {
          answer: inputValue,
          ask_id: askId,
        },
      });
      setAskId(null);
    } else {
      sendQuery(inputValue);
    }
  };

  // const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
  //   if (e.key === 'Enter' && !e.shiftKey && !isModelResponding) {
  //     e.preventDefault();
  //     handleSendMessage();
  //   }
  // };

  const downloadReport = (content: any, contentType: "html" | "svg" | "markdown") => {
    try {
      const data = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
      
      const contentTypeMap = {
        html: { extension: '.html', mimeType: 'text/html' },
        svg: { extension: '.svg', mimeType: 'image/svg+xml' },
        markdown: { extension: '.md', mimeType: 'text/markdown' }
      };
      const { extension: fileExtension, mimeType } = contentTypeMap[contentType] || { extension: '.md', mimeType: 'text/markdown' };

      const blob = new Blob([data], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report-${new Date().toISOString().slice(0, 10)}${fileExtension}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  };

  // Handle scroll events to detect user's scroll position
  const handleScroll = useCallback(() => {
    if (!mainContainerRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = document.documentElement;
    const isAtBottom = scrollHeight - (scrollTop + clientHeight) < 100; // 100px threshold
    
    if (isAtBottom !== isUserAtBottom) {
      setIsUserAtBottom(isAtBottom);
      console.log('isAtBottom', isAtBottom);
    }
  }, [isUserAtBottom]);

  // Add scroll event listener to window
  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  const scrollToBottom = () => {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: 'smooth'
    });
  };

  // Auto-scroll to bottom when messages change and user is at bottom
  useEffect(() => {
    if (isUserAtBottom) {
      scrollToBottom();
    }
  }, [messages, isUserAtBottom]);

  // Function to scroll to a specific message
  const scrollToMessage = useCallback((id: number) => {
    const messageElement = document.getElementById(`message-${id}`);
    if (messageElement) {
      messageElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      // Add highlight effect
      messageElement.classList.add('highlight-message');
      setTimeout(() => {
        messageElement.classList.remove('highlight-message');
      }, 2000);
    }
  }, []);

  const handleOpenSettings = () => setSettingsOpen(true);
  const handleCloseSettings = () => setSettingsOpen(false);
  
  const handleSaveSettings = () => {
    localStorage.setItem('wsUrl', wsUrl);
    handleCloseSettings();
    // Reload the page to reconnect with the new WebSocket URL
    window.location.reload();
  };

  const handleConfigSelect = (config: string) => {
    let request: UserRequest = {
      type: 'switch_agent', 
      content: { config_file: config } 
    };
    sendRequest(request);
  };

  const handleAddConfig = () => {
    let request: UserRequest = {
      type: 'gen_agent', 
      content: null 
    };
    sendRequest(request);
  };

  const [isMobileView, setIsMobileView] = useState(window.innerWidth <= 768);

  // Handle window resize for responsive behavior
  useEffect(() => {
    const handleResize = () => {
      const isMobile = window.innerWidth <= 768;
      setIsMobileView(isMobile);
      
      if (!isMobile) {
        setIsSidebarOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const toggleSidebar = () => {
    console.log("toggleSidebar", isSidebarOpen, !isSidebarOpen);
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="app-container">
      {isMobileView && (
        <HamburgerButton 
          isOpen={isSidebarOpen} 
          onClick={toggleSidebar} 
        />
      )}
      <SideBar 
        isOpen={isMobileView ? isSidebarOpen : true}
        onClose={() => {setTimeout(() => {setIsSidebarOpen(false)}, 100)}}
        messages={messages}
        onNavigate={scrollToMessage}
        currentConfig={currentConfig}
        agentType={agentType}
        subAgents={subAgents}
        onConfigSelect={handleConfigSelect}
        handleAddConfig={handleAddConfig}
        getConfigList={getConfigList}
        availableConfigs={availableConfigs}
        showNewChatButton={showNewChatButton}
        showAgentConfigs={showAgentConfigs}
      />
      
      <div className="main-content">
        <button 
          className="settings-button"
          onClick={handleOpenSettings}
          title={t('app.settingsButtonTitle')}
        >
          <i className="fas fa-cog"></i>
        </button>
        
        {settingsOpen && (
          <div className="modal-overlay" onClick={handleCloseSettings}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
              <h3>{t('app.settingsModalTitle')}</h3>
              <div className="form-group">
                <label htmlFor="wsUrl">{t('app.websocketUrlLabel')}</label>
                <input
                  id="wsUrl"
                  type="text"
                  value={wsUrl}
                  onChange={(e) => setWsUrl(e.target.value)}
                  placeholder="ws://localhost:8848/ws"
                />
              </div>
              <div className="form-group">
                <label htmlFor="languageSelect">{t('app.languageLabel')}</label>
                <select
                  id="languageSelect"
                  value={i18n.language?.startsWith('zh') ? 'zh' : 'en'}
                  onChange={(e) => i18n.changeLanguage(e.target.value)}
                >
                  <option value="en">{t('language.en')}</option>
                  <option value="zh">{t('language.zh')}</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="themeSelect">{t('app.themeLabel')}</label>
                <select
                  id="themeSelect"
                  value={theme}
                  onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}
                >
                  <option value="light">{t('app.themeLight')}</option>
                  <option value="dark">{t('app.themeDark')}</option>
                </select>
              </div>
              <div className="modal-actions">
                <button onClick={handleCloseSettings}>
                  {t('app.cancel')}
                </button>
                <button onClick={handleSaveSettings} className="primary">
                  {t('app.save')}
                </button>
              </div>
            </div>
          </div>
        )}
        <div className="chat-container" ref={mainContainerRef}>
          <div className="header-container">
            <h1 className="chat-title">
              <img className="title-logo" src={logoSquareUrl} alt="logo" />
              Youtu Agent
            </h1>
          </div>

          <div className="chat-messages">
            {messages.map((message, index) => {
              // Check if previous message is from the same sender
              const showSender = index === 0 ||
                messages[index - 1].sender !== message.sender ||
                message.sender === 'user';

              return (
                <div key={message.id} id={`message-${message.id}`}>
                  <MessageComponent
                    message={message}
                    messageId={message.id.toString()}
                    showSender={showSender}
                    onDownloadReport={message.type === 'report' ? (content, contentType) => {
                      downloadReport(content, contentType);
                    } : undefined}
                  />
                </div>
              );
            })}
            
            {isModelResponding && (
              <div className="respond-placeholder">
                <i className="breathing-circle fa fa-circle"></i>
              </div>
            )}
            
            {!hideExampleQuery && exampleQuery.length > 0 && (
              <div className="example-queries">
                <div className="example-queries-title">{t('app.tryAsking')}</div>
                {exampleQuery.map((query, idx) => (
                  <div key={idx} className="example-query" onClick={() => {
                    setInputValue(query);
                    setHideExampleQuery(true);
                  }}>
                    {query}
                  </div>
                ))}
              </div>
            )}
          </div>
          

          <ChatInput
            inputValue={inputValue}
            onInputChange={setInputValue}
            onSendMessage={handleSendMessage}
            isModelResponding={isModelResponding}
            inputRef={inputRef}
            currentConfig={currentConfig}
            onConfigSelect={handleConfigSelect}
            handleAddConfig={handleAddConfig}
            chatInputLoadingState={chatInputLoadingState}
            setChatInputLoadingState={setChatInputLoadingState}
            availableConfigs={availableConfigs}
            getConfigList={getConfigList}
          />

          {/* Footer with pic.png */}
          <footer className="app-footer">
            <div className="p-4">
              <img src={logoUrl} alt="Footer Logo" className="footer-logo" />
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
};

export default App;
