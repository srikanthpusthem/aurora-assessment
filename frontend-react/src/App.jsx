import { useState, useCallback } from 'react';
import { Header } from './components/Header/Header';
import { QuestionInput } from './components/QuestionInput/QuestionInput';
import { Answer } from './components/Answer/Answer';
import { Loading } from './components/Loading/Loading';
import { Error } from './components/Error/Error';
import { History } from './components/History/History';
import { useAPI } from './hooks/useAPI';
import { useHistory } from './hooks/useHistory';
import { useHealthCheck } from './hooks/useHealthCheck';
import './App.css';

function App() {
  const [answer, setAnswer] = useState(null);
  const [answerTimestamp, setAnswerTimestamp] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);

  const { askQuestion, isLoading, error, clearError } = useAPI();
  const { history, addToHistory, clearHistory } = useHistory();
  const { status, isIndexReady } = useHealthCheck();

  const handleSubmit = useCallback(async (question) => {
    setCurrentQuestion(question);
    setAnswer(null);
    setAnswerTimestamp(null);
    clearError();

    try {
      const response = await askQuestion(question);
      const answerText = response.answer;
      setAnswer(answerText);
      setAnswerTimestamp(new Date().toISOString());
      addToHistory(question, answerText);
    } catch (err) {
      // Error is handled by useAPI hook
      console.error('Error asking question:', err);
    }
  }, [askQuestion, addToHistory, clearError]);

  const handleRetry = useCallback(() => {
    if (currentQuestion) {
      handleSubmit(currentQuestion);
    }
  }, [currentQuestion, handleSubmit]);

  const handleSelectFromHistory = useCallback((question) => {
    handleSubmit(question);
  }, [handleSubmit]);

  const handleClearHistory = useCallback(() => {
    if (window.confirm('Are you sure you want to clear your question history?')) {
      clearHistory();
    }
  }, [clearHistory]);

  return (
    <div className="App">
      <Header status={status} isIndexReady={isIndexReady} />
      
      <main className="main">
        <div className="container">
          {/* Hero Section */}
          <section className="hero">
            <h2 className="hero-title">Ask Anything</h2>
            <p className="hero-subtitle">
              Get intelligent answers from Aurora's message database
            </p>
          </section>

          {/* Question Input */}
          <QuestionInput onSubmit={handleSubmit} isLoading={isLoading} />

          {/* Loading State */}
          {isLoading && <Loading />}

          {/* Error State */}
          {error && <Error message={error} onRetry={handleRetry} />}

          {/* Answer */}
          {answer && <Answer answer={answer} timestamp={answerTimestamp} />}

          {/* History */}
          {history.length > 0 && (
            <History
              history={history}
              onSelectQuestion={handleSelectFromHistory}
              onClearHistory={handleClearHistory}
            />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p className="footer-text">
            Powered by <strong>Aurora QA System</strong> |{' '}
            <span>{isIndexReady ? 'Index ready' : 'Index building...'}</span>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
