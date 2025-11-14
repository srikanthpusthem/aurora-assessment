import { CONFIG } from '../utils/constants';

const STORAGE_KEY = 'auroraQAHistory';

/**
 * Storage Service for managing localStorage
 */
class StorageService {
  /**
   * Load history from localStorage
   */
  loadHistory() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Failed to load history:', error);
      return [];
    }
  }

  /**
   * Save history to localStorage
   */
  saveHistory(history) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (error) {
      console.error('Failed to save history:', error);
    }
  }

  /**
   * Add item to history
   */
  addToHistory(history, question, answer) {
    // Remove if already exists
    let newHistory = history.filter(item => item.question !== question);
    
    // Add to beginning
    newHistory.unshift({
      question,
      answer,
      timestamp: new Date().toISOString(),
    });
    
    // Limit history size
    if (newHistory.length > CONFIG.MAX_HISTORY) {
      newHistory = newHistory.slice(0, CONFIG.MAX_HISTORY);
    }
    
    this.saveHistory(newHistory);
    return newHistory;
  }

  /**
   * Clear history
   */
  clearHistory() {
    try {
      localStorage.removeItem(STORAGE_KEY);
      return [];
    } catch (error) {
      console.error('Failed to clear history:', error);
      return [];
    }
  }
}

export const storageService = new StorageService();

