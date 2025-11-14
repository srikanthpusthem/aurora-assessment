# Logging Guide

## Overview

The Aurora QA System includes comprehensive logging that outputs to both the console and a log file for easy debugging and learning.

## Log File Location

By default, logs are written to: `logs/aurora_qa.log`

## Configuration

You can configure logging in your `.env` file:

```env
# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Log file path (empty = console only)
LOG_FILE=logs/aurora_qa.log

# Log rotation settings (optional)
LOG_MAX_BYTES=10485760  # 10MB per file
LOG_BACKUP_COUNT=5      # Keep 5 backup files
```

## Viewing Logs

### Real-time Console Output
Logs appear in your terminal when running the application.

### View Log File

```bash
# View logs in real-time
tail -f logs/aurora_qa.log

# View last 50 lines
tail -n 50 logs/aurora_qa.log

# View entire log file
cat logs/aurora_qa.log

# View with pagination
less logs/aurora_qa.log
```

### Filter Logs by Component

```bash
# View only QA engine logs
grep "\[QA\]" logs/aurora_qa.log

# View only retrieval logs
grep "\[RETRIEVAL\]" logs/aurora_qa.log

# View only LLM logs
grep "\[LLM\]" logs/aurora_qa.log

# View only embedding logs
grep "\[EMBEDDING\]" logs/aurora_qa.log

# View only startup logs
grep "\[STARTUP\]" logs/aurora_qa.log

# View only request/response logs
grep "\[REQUEST\]\|\[RESPONSE\]" logs/aurora_qa.log
```

### Filter by Log Level

```bash
# View only errors
grep "ERROR" logs/aurora_qa.log

# View warnings and errors
grep -E "WARNING|ERROR" logs/aurora_qa.log

# View debug messages (if LOG_LEVEL=DEBUG)
grep "DEBUG" logs/aurora_qa.log
```

### Search for Specific Content

```bash
# Search for a specific question
grep "Where is Sophia" logs/aurora_qa.log

# Search for API calls
grep "API call" logs/aurora_qa.log

# Search for timing information
grep "in.*s" logs/aurora_qa.log
```

### Advanced Filtering

```bash
# View logs from a specific time range
grep "2024-01-15 10:" logs/aurora_qa.log

# View logs with context (3 lines before and after)
grep -C 3 "ERROR" logs/aurora_qa.log

# Count occurrences
grep -c "\[QA\]" logs/aurora_qa.log

# View unique log tags
grep -o "\[.*\]" logs/aurora_qa.log | sort | uniq
```

## Log Format

Each log entry follows this format:

```
[YYYY-MM-DD HH:MM:SS] LEVEL logger_name - [TAG] message
```

Example:
```
[2024-01-15 10:30:45] INFO app.main - [STARTUP] Starting Aurora QA System...
[2024-01-15 10:31:00] INFO app.api.ask - [REQUEST] POST /api/ask - Question: "Where is Sophia going?"
[2024-01-15 10:31:01] INFO app.services.qa_engine - [QA] Processing question: "Where is Sophia going?"
```

## Log Tags

The system uses consistent tags to identify different operations:

- `[STARTUP]` - Application startup and initialization
- `[STEP X/4]` - Startup steps (1=ingestion, 2=embedding, 3=indexing, 4=ready)
- `[REQUEST]` - Incoming API requests
- `[RESPONSE]` - API responses
- `[QA]` - QA engine orchestration
- `[RETRIEVAL]` - Message retrieval operations
- `[EMBEDDING]` - Text embedding operations
- `[LLM]` - LLM extraction operations
- `[SEARCH]` - FAISS index search operations

## Log Rotation

Logs are automatically rotated when they reach 10MB. The system keeps 5 backup files:
- `logs/aurora_qa.log` (current)
- `logs/aurora_qa.log.1` (previous)
- `logs/aurora_qa.log.2` (older)
- etc.

## Disable File Logging

To disable file logging and only use console output, set in `.env`:

```env
LOG_FILE=
```

Or set it to an empty string in the environment.

## Debug Mode

For maximum verbosity, set:

```env
LOG_LEVEL=DEBUG
```

This will show:
- All data transformations
- Sample values and statistics
- Full API request/response details
- Detailed timing information
- Complete error traces

## Tips

1. **Use `tail -f`** to watch logs in real-time while testing
2. **Filter by tag** to focus on specific components
3. **Search for errors** when debugging issues
4. **Use DEBUG level** when learning how the system works
5. **Check log rotation** if logs seem incomplete (check backup files)

## Example Workflow

```bash
# 1. Start the application (logs go to both console and file)
uvicorn app.main:app --reload

# 2. In another terminal, watch logs in real-time
tail -f logs/aurora_qa.log

# 3. Filter for specific operations
grep "\[QA\]" logs/aurora_qa.log | tail -20

# 4. Check for errors
grep "ERROR" logs/aurora_qa.log
```

