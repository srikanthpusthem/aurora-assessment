import { EXAMPLE_QUESTIONS } from '../../utils/constants';
import './ExampleChips.css';

export function ExampleChips({ onSelect }) {
  return (
    <div className="examples-section">
      <p className="examples-label">Try asking:</p>
      <div className="examples-grid">
        {EXAMPLE_QUESTIONS.map((question, index) => (
          <button
            key={index}
            className="example-chip"
            onClick={() => onSelect(question)}
            type="button"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

