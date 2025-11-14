import { StatusIndicator } from './StatusIndicator';
import './Header.css';

export function Header({ status, isIndexReady }) {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <h1 className="logo">
            <span className="logo-icon">✨</span>
            Aurora QA
          </h1>
          <StatusIndicator status={status} isIndexReady={isIndexReady} />
        </div>
      </div>
    </header>
  );
}

