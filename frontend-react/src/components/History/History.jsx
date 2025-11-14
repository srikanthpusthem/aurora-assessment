import './History.css';

export function History({ history, onSelectQuestion, onClearHistory }) {
  if (!history || history.length === 0) return null;

  return (
    <section className="history-section fade-in">
      <div className="history-card">
        <div className="history-header">
          <h3 className="history-title">Recent Questions</h3>
          <button
            className="clear-history-btn"
            onClick={onClearHistory}
            type="button"
          >
            Clear
          </button>
        </div>
        <div className="history-list">
          {history.map((item, index) => (
            <div
              key={index}
              className="history-item"
              onClick={() => onSelectQuestion(item.question)}
            >
              <div className="history-question">{item.question}</div>
              <div className="history-answer">{item.answer}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

