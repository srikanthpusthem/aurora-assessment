import './StatusIndicator.css';

export function StatusIndicator({ status, isIndexReady }) {
  const getStatusClass = () => {
    if (status === 'connected') return 'connected';
    if (status === 'error') return 'error';
    return '';
  };

  const getStatusText = () => {
    if (status === 'connected') {
      return isIndexReady ? 'Connected' : 'Connecting...';
    }
    if (status === 'error') return 'Disconnected';
    return 'Checking...';
  };

  return (
    <div className={`status-indicator ${getStatusClass()}`}>
      <span className="status-dot"></span>
      <span className="status-text">{getStatusText()}</span>
    </div>
  );
}

