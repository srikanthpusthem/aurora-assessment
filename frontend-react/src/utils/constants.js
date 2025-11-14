export const CONFIG = {
  API_BASE_URL: import.meta.env.VITE_API_URL || 
    (window.location.origin.includes('localhost') 
      ? 'http://localhost:8000' 
      : window.location.origin),
  API_ENDPOINT: '/api/ask',
  HEALTH_ENDPOINT: '/health',
  TIMEOUT: 60000, // 60 seconds
  MAX_HISTORY: 10
};

export const EXAMPLE_QUESTIONS = [
  'Where is Sophia going?',
  'What did someone request about a private jet?',
  'Tell me about dinner reservations',
  'What travel plans are mentioned?'
];

