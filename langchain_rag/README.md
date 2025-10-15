# Enhanced RAG Chatbot with Memory and PostgreSQL

A sophisticated RAG (Retrieval-Augmented Generation) chatbot built with FastAPI, LangChain, and Google Gemini that includes persistent memory storage using PostgreSQL.

## Features

- **RAG-based Question Answering**: Uses FAISS vector store with Gemini embeddings for document retrieval
- **Persistent Memory**: Stores chat history in PostgreSQL database
- **Session Management**: Multi-user support with session-based conversations
- **LangChain Integration**: Uses LangChain's PostgresChatMessageHistory for memory management
- **Rate Limiting**: Built-in API rate limiting for production use
- **RESTful API**: Clean FastAPI endpoints for easy integration

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Google Gemini API key

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL:**
   - Install PostgreSQL on your system
   - Start the PostgreSQL service
   - Create a user and database (or use the setup script)

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   # Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here

   # PostgreSQL Database Configuration
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=rag_chatbot
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password_here
   ```

5. **Set up the database:**
   ```bash
   python setup_database.py
   ```

6. **Place your PDF document:**
   Update the `PDF_PATH` in `rag_gemini.py` to point to your PDF file:
   ```python
   PDF_PATH = Path("path/to/your/document.pdf")
   ```

## Usage

1. **Start the application:**
   ```bash
   uvicorn rag_gemini:app --reload
   ```

2. **Access the API documentation:**
   Open your browser and go to `http://localhost:8000/docs`

## API Endpoints

### Core Chat Functionality

- **POST `/ask`** - Ask a question to the chatbot
  ```json
  {
    "question": "What is time management?",
    "session_id": "optional-session-id",
    "user_id": "optional-user-id"
  }
  ```

### Session Management

- **POST `/sessions`** - Create a new chat session
- **GET `/sessions`** - List all sessions (with optional user_id filter)
- **GET `/sessions/{session_id}/history`** - Get chat history for a session
- **DELETE `/sessions/{session_id}`** - Delete a session and its history

### Utility Endpoints

- **GET `/`** - Welcome message
- **GET `/health`** - Health check
- **POST `/rebuild-index`** - Rebuild the FAISS vector index

## Example Usage

### Starting a New Conversation

```bash
curl -X POST "http://localhost:8000/sessions" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Asking Questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key principles of time management?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user123"
  }'
```

Response:
```json
{
  "question": "What are the key principles of time management?",
  "answer": "Based on the document, the key principles include...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:31:00Z"
}
```

### Getting Chat History

```bash
curl -X GET "http://localhost:8000/sessions/550e8400-e29b-41d4-a716-446655440000/history"
```

## Database Schema

The application creates two main tables:

### `chat_sessions`
- `session_id` (String, Primary Key)
- `created_at` (DateTime)
- `last_activity` (DateTime)
- `user_id` (String, Optional)

### `chat_messages`
- `id` (Integer, Primary Key, Auto-increment)
- `session_id` (String, Foreign Key)
- `message_type` (String: 'human' or 'ai')
- `content` (Text)
- `timestamp` (DateTime)
- `metadata` (Text, Optional JSON)

## Memory Features

The chatbot includes several memory-related features:

1. **Conversation Context**: Recent conversation history is included in prompts for better context-aware responses
2. **Session Persistence**: Chat sessions persist across application restarts
3. **Multi-user Support**: Different users can have separate conversation histories
4. **Configurable Memory**: Adjust the number of recent messages used for context

## Configuration

Key configuration options in the code:

- `max_messages` in `get_conversation_context()`: Number of recent messages to include in context (default: 10)
- `top_k` in `answer_question()`: Number of document chunks to retrieve (default: 4)
- `RATE_LIMIT`: Maximum requests per time window (default: 500)
- `TIME_WINDOW`: Rate limit time window in seconds (default: 60)

## Troubleshooting

### Common Issues

1. **Database Connection Error**: Ensure PostgreSQL is running and credentials are correct
2. **PDF Not Found**: Update the `PDF_PATH` variable to point to your PDF file
3. **Gemini API Error**: Verify your API key is valid and has sufficient quota
4. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Logs

The application provides detailed logging for:
- Database operations
- Vector index loading/building
- API requests and responses
- Error conditions

## Development

To extend the application:

1. **Add new memory types**: Extend the `ChatMessage` model and related functions
2. **Customize prompts**: Modify the prompt templates in `answer_question()`
3. **Add new endpoints**: Follow the existing FastAPI patterns
4. **Enhance vector search**: Modify the `search_similar()` function

## License

This project is open source and available under the MIT License.
