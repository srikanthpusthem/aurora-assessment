import { useState, useEffect, useCallback } from 'react';
import { storageService } from '../services/storage';

/**
 * Custom hook for managing question history
 */
export function useHistory() {
  const [history, setHistory] = useState(() => storageService.loadHistory());

  useEffect(() => {
    // Sync with localStorage when history changes
    storageService.saveHistory(history);
  }, [history]);

  const addToHistory = useCallback((question, answer) => {
    setHistory(prev => storageService.addToHistory(prev, question, answer));
  }, []);

  const clearHistory = useCallback(() => {
    const cleared = storageService.clearHistory();
    setHistory(cleared);
  }, []);

  return {
    history,
    addToHistory,
    clearHistory,
  };
}

