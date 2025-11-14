# Aurora QA System - Frontend

A sleek, modern web frontend for the Aurora QA System.

## Features

- **Modern UI**: Glassmorphism design with gradient backgrounds and smooth animations
- **Real-time QA**: Ask questions and get instant answers from the backend
- **Question History**: Automatically saves your recent questions (localStorage)
- **Example Questions**: Clickable suggestions to get started quickly
- **Copy to Clipboard**: Easy answer copying with visual feedback
- **Keyboard Shortcuts**: 
  - `Ctrl/Cmd + Enter`: Submit question
  - `Escape`: Clear input
- **Health Monitoring**: Real-time backend connection status
- **Responsive Design**: Works beautifully on desktop, tablet, and mobile

## File Structure

```
frontend/
├── index.html          # Main HTML file
├── styles/
│   └── main.css       # All styling (gradients, animations, responsive)
├── scripts/
│   └── app.js         # Main JavaScript (API integration, state management)
└── README.md          # This file
```

## Usage

### Local Development

1. Make sure the backend is running on `http://localhost:8000`
2. Open `index.html` in a web browser
   - Or use a local server: `python -m http.server 8080` (then visit `http://localhost:8080`)
   - Or use VS Code Live Server extension

### Integration with FastAPI

To serve the frontend from FastAPI, add this to `app/main.py`:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")
```

## Configuration

The frontend automatically detects the API URL:
- If running on `localhost`, uses `http://localhost:8000`
- Otherwise, uses the current origin

To customize, edit `CONFIG.API_BASE_URL` in `scripts/app.js`.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

Requires modern browser features:
- ES6+ JavaScript
- Fetch API
- CSS Grid & Flexbox
- LocalStorage

## Design

- **Color Scheme**: Dark theme with purple/blue gradients
- **Typography**: Inter font family
- **Layout**: Centered, max-width container
- **Animations**: Smooth fade-in, slide-up, and pulse effects
- **Glassmorphism**: Frosted glass effect on cards

## Features in Detail

### Question Input
- Large, prominent textarea
- Auto-focus on page load
- Enter key support
- Disabled state during loading

### Answer Display
- Animated reveal
- Copy button with feedback
- Timestamp
- Smooth scroll to answer

### Error Handling
- User-friendly error messages
- Retry button
- Network error detection
- Timeout handling

### History
- Stores last 10 questions
- Click to re-ask
- Clear history option
- Persistent across sessions

## Keyboard Shortcuts

- `Ctrl/Cmd + Enter`: Submit question
- `Escape`: Clear input field

## API Integration

### Endpoints Used

1. **POST /api/ask**
   ```json
   Request: {"question": "string"}
   Response: {"answer": "string"}
   ```

2. **GET /health**
   ```json
   Response: {"status": "healthy", "index_ready": true}
   ```

### Error Handling

- **400**: Validation errors (empty question, etc.)
- **500**: Server errors
- **Network**: Connection issues
- **Timeout**: 60-second timeout with retry option

## Customization

### Colors

Edit CSS variables in `styles/main.css`:

```css
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --bg-gradient: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 50%, #1a1a2e 100%);
    /* ... */
}
```

### API URL

Edit `CONFIG.API_BASE_URL` in `scripts/app.js`.

### History Limit

Edit `CONFIG.MAX_HISTORY` in `scripts/app.js` (default: 10).

## License

Part of the Aurora QA System project.

