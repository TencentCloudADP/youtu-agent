import { useCallback, useRef } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';

import type { UserRequest } from '../types/requests';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface UseChatWebSocketOptions {
  onConnectionChange?: (status: ConnectionStatus) => void;
}

export const useChatWebSocket = (ws_url: string, { onConnectionChange }: UseChatWebSocketOptions = {}) => {
  const wasConnected = useRef(false);
  
  const { sendMessage, lastMessage, readyState, getWebSocket } = useWebSocket(ws_url, {
    onOpen: () => {
      wasConnected.current = true;
      onConnectionChange?.('connected');
    },
    onError: (event) => {
      console.error('WebSocket error:', event);
      onConnectionChange?.('error');
    },
    onClose: (event) => {
      console.log('WebSocket connection closed:', event);
      if (wasConnected.current) {
        onConnectionChange?.('disconnected');
      }
    },
    shouldReconnect: (_) => true, // Will attempt to reconnect on all close events
    reconnectAttempts: 1, // Number of reconnect attempts
    reconnectInterval: 3000, // Wait 3 seconds between reconnection attempts
  });

  // Send a query to the server
  const sendQuery = useCallback((query: string) => {
    if (readyState === ReadyState.OPEN) {
      const message = JSON.stringify({
        type: 'query',
        content: {
          query: query
        }
      });
      sendMessage(message);
      return true;
    }
    console.error('WebSocket is not connected');
    return false;
  }, [readyState, sendMessage]);

  const sendRequest = useCallback((request: UserRequest) => {
    if (readyState === ReadyState.OPEN) {
      const message = JSON.stringify(request);
      sendMessage(message);
      return true;
    }
    console.error('WebSocket is not connected');
    return false;
  }, [readyState, sendMessage]);

  return {
    sendQuery,
    sendRequest,
    lastMessage,
    readyState,
    getWebSocket,
  };
};

export default useChatWebSocket;
