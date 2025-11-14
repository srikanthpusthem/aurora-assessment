import { useState, useCallback } from 'react';
import { apiService } from '../services/api';

/**
 * Custom hook for API interactions
 */
export function useAPI() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const askQuestion = useCallback(async (question) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await apiService.askQuestion(question);
      return response;
    } catch (err) {
      const errorMessage = err.message || 'An error occurred while processing your question.';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    askQuestion,
    isLoading,
    error,
    clearError,
  };
}

