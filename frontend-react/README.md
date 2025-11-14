# Aurora QA System - React Frontend

A modern, sleek React frontend for the Aurora QA System built with Vite.

## Features

- **React 18** with modern hooks
- **Component-based architecture** for maintainability
- **Custom hooks** for API, history, and health checks
- **Smooth animations** and transitions
- **Responsive design** (mobile, tablet, desktop)
- **Keyboard shortcuts** (Ctrl+Enter to submit, Escape to clear)
- **Question history** with localStorage persistence
- **Copy to clipboard** functionality
- **Real-time health monitoring**
- **Glassmorphism design** with gradient backgrounds

## Technology Stack

- **React 18**: Latest React with hooks
- **Vite**: Fast build tool and dev server
- **CSS3**: Modern styling with animations
- **Custom Hooks**: useAPI, useHistory, useHealthCheck

## Project Structure

```
frontend-react/
├── public/              # Static assets
├── src/
│   ├── components/      # React components
│   │   ├── Header/
│   │   ├── QuestionInput/
│   │   ├── Answer/
│   │   ├── Loading/
│   │   ├── Error/
│   │   └── History/
│   ├── hooks/           # Custom React hooks
│   ├── services/        # API and storage services
│   ├── utils/           # Constants and utilities
│   ├── App.jsx          # Main App component
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── package.json
├── vite.config.js
└── README.md
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend running on `http://localhost:8000`

### Installation

```bash
cd frontend-react
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173` (or the port Vite assigns).

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory, ready to be served statically.

### Preview Production Build

```bash
npm run preview
```

## Configuration

### API URL

The frontend automatically detects the API URL:
- If running on `localhost`, uses `http://localhost:8000`
- Otherwise, uses the current origin

To customize, create a `.env` file:

```env
VITE_API_URL=http://your-api-url:8000
```

Or edit `src/utils/constants.js`:

```javascript
export const CONFIG = {
  API_BASE_URL: 'http://your-api-url:8000',
  // ...
};
```

## Components

### Header
- Logo and branding
- Status indicator (connection status)

### QuestionInput
- Textarea for questions
- Submit button
- Example question chips
- Keyboard shortcuts

### Answer
- Answer display with animations
- Copy to clipboard button
- Timestamp

### Loading
- Spinner animation
- "Thinking..." message

### Error
- Error message display
- Retry button

### History
- Recent questions list
- Click to re-ask
- Clear history option

## Custom Hooks

### useAPI
Manages API calls and loading/error states.

```javascript
const { askQuestion, isLoading, error, clearError } = useAPI();
```

### useHistory
Manages question history with localStorage.

```javascript
const { history, addToHistory, clearHistory } = useHistory();
```

### useHealthCheck
Monitors backend health status.

```javascript
const { status, isIndexReady, isConnected } = useHealthCheck();
```

## Services

### API Service (`services/api.js`)
- Centralized API configuration
- Error handling
- Timeout management

### Storage Service (`services/storage.js`)
- localStorage abstraction
- History management

## Features

### Keyboard Shortcuts
- `Ctrl/Cmd + Enter`: Submit question
- `Escape`: Clear input field

### Question History
- Automatically saves last 10 questions
- Persistent across sessions (localStorage)
- Click to re-ask questions

### Copy to Clipboard
- One-click answer copying
- Visual feedback when copied

### Health Monitoring
- Real-time connection status
- Periodic health checks (every 30 seconds)
- Index ready status

## Styling

The app uses global CSS with CSS custom properties for theming. All styles are in `src/index.css` with component-specific styles in their respective CSS files.

### Color Scheme
- Dark theme with gradient backgrounds
- Purple/blue primary gradient
- Glassmorphism effects

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Integration with FastAPI

To serve the React app from FastAPI:

1. Build the React app: `npm run build`
2. Copy `dist/` contents to a static directory
3. Serve from FastAPI:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="frontend-react/dist"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("frontend-react/dist/index.html")
```

## Development Tips

- Hot Module Replacement (HMR) is enabled for fast development
- Use React DevTools for debugging
- Check browser console for API errors
- Network tab shows all API requests

## License

Part of the Aurora QA System project.
