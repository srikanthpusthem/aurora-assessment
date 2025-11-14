import { useEffect, useRef } from 'react';
import { CopyButton } from './CopyButton';
import './Answer.css';

export function Answer({ answer, timestamp }) {
  const answerRef = useRef(null);

  useEffect(() => {
    // Scroll to answer when it appears
    if (answerRef.current && answer) {
      setTimeout(() => {
        answerRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
        });
      }, 100);
    }
  }, [answer]);

  if (!answer) return null;

  return (
    <section className="answer-section slide-up" ref={answerRef}>
      <div className="answer-card">
        <div className="answer-header">
          <h3 className="answer-title">Answer</h3>
          <CopyButton text={answer} />
        </div>
        <div className="answer-content">{answer}</div>
        {timestamp && (
          <div className="answer-footer">
            <span className="answer-timestamp">
              Answered at {new Date(timestamp).toLocaleTimeString()}
            </span>
          </div>
        )}
      </div>
    </section>
  );
}

