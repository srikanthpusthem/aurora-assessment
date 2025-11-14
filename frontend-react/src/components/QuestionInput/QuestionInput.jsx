import { useState, useRef, useEffect } from 'react';
import { ExampleChips } from './ExampleChips';
import './QuestionInput.css';

export function QuestionInput({ onSubmit, isLoading }) {
  const [question, setQuestion] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    // Auto-focus on mount
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmedQuestion = question.trim();
    if (trimmedQuestion && !isLoading) {
      onSubmit(trimmedQuestion);
    }
  };

  const handleExampleSelect = (exampleQuestion) => {
    setQuestion(exampleQuestion);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
    // Auto-submit after a short delay
    setTimeout(() => {
      onSubmit(exampleQuestion);
    }, 100);
  };

  const handleKeyDown = (e) => {
    // Ctrl+Enter or Cmd+Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit(e);
    }
    // Escape to clear
    if (e.key === 'Escape') {
      setQuestion('');
      textareaRef.current?.blur();
    }
  };

  return (
    <section className="input-section">
      <div className="input-card">
        <form className="question-form" onSubmit={handleSubmit}>
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              className="question-input"
              placeholder="Type your question here... (e.g., Where is Sophia going?)"
              rows="3"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              required
              aria-label="Question input"
            />
            <button
              type="submit"
              className="submit-btn"
              disabled={isLoading || !question.trim()}
              aria-label="Submit question"
            >
              <span className="submit-icon">→</span>
            </button>
          </div>
        </form>
      </div>
      <ExampleChips onSelect={handleExampleSelect} />
    </section>
  );
}

