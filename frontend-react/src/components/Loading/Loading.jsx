import './Loading.css';

export function Loading() {
  return (
    <section className="loading-section fade-in">
      <div className="loading-card">
        <div className="spinner"></div>
        <p className="loading-text">Thinking...</p>
      </div>
    </section>
  );
}

