import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

/**
 * Custom hook for health check monitoring
 */
export function useHealthCheck() {
  const [isConnected, setIsConnected] = useState(false);
  const [isIndexReady, setIsIndexReady] = useState(false);
  const [status, setStatus] = useState('checking'); // 'checking' | 'connected' | 'error'

  const checkHealth = useCallback(async () => {
    try {
      const data = await apiService.checkHealth();
      setIsConnected(true);
      setIsIndexReady(data.index_ready || false);
      setStatus('connected');
      return data;
    } catch (error) {
      setIsConnected(false);
      setIsIndexReady(false);
      setStatus('error');
      return null;
    }
  }, []);

  useEffect(() => {
    // Initial check
    checkHealth();

    // Periodic checks every 30 seconds
    const interval = setInterval(checkHealth, 30000);

    return () => clearInterval(interval);
  }, [checkHealth]);

  return {
    isConnected,
    isIndexReady,
    status,
    checkHealth,
  };
}

