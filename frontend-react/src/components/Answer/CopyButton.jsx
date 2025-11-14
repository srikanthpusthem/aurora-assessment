import { useState } from 'react';
import './CopyButton.css';

export function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
      // Fallback: select text
      const range = document.createRange();
      const selection = window.getSelection();
      const answerElement = document.querySelector('.answer-content');
      if (answerElement) {
        range.selectNodeContents(answerElement);
        selection.removeAllRanges();
        selection.addRange(range);
      }
    }
  };

  return (
    <button
      className={`copy-btn ${copied ? 'copied' : ''}`}
      onClick={handleCopy}
      aria-label="Copy answer"
      type="button"
    >
      <span className="copy-icon">📋</span>
      <span className="copy-text">{copied ? 'Copied!' : 'Copy'}</span>
    </button>
  );
}

