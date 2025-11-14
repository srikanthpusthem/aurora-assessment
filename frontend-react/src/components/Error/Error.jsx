import { useEffect, useRef } from 'react';
import './Error.css';

export function Error({ message, onRetry }) {
  const errorRef = useRef(null);

  useEffect(() => {
    // Scroll to error when it appears
    if (errorRef.current && message) {
      setTimeout(() => {
        errorRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
        });
      }, 100);
    }
  }, [message]);

  if (!message) return null;

  return (
    <section className="error-section shake" ref={errorRef}>
      <div className="error-card">
        <span className="error-icon">⚠️</span>
        <p className="error-message">{message}</p>
        {onRetry && (
          <button className="retry-btn" onClick={onRetry} type="button">
            Try Again
          </button>
        )}
      </div>
    </section>
  );
}

