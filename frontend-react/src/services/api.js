import { CONFIG } from '../utils/constants';

/**
 * API Service for communicating with the backend
 */
class APIService {
  constructor() {
    this.baseURL = CONFIG.API_BASE_URL;
    this.timeout = CONFIG.TIMEOUT;
  }

  /**
   * Create an AbortController with timeout
   */
  createTimeoutController() {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    return { controller, timeoutId };
  }

  /**
   * Generic fetch wrapper with error handling
   */
  async fetch(endpoint, options = {}) {
    const { controller, timeoutId } = this.createTimeoutController();
    
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new Error('Request timed out. Please try again.');
      }
      
      if (error.message.includes('Failed to fetch')) {
        throw new Error('Unable to connect to the server. Please check your connection.');
      }
      
      throw error;
    }
  }

  /**
   * Ask a question
   */
  async askQuestion(question) {
    return this.fetch(CONFIG.API_ENDPOINT, {
      method: 'POST',
      body: JSON.stringify({ question }),
    });
  }

  /**
   * Check backend health
   */
  async checkHealth() {
    return this.fetch(CONFIG.HEALTH_ENDPOINT, {
      method: 'GET',
    });
  }
}

export const apiService = new APIService();

