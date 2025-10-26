# Local Command-Line Chatbot using Hugging Face

A fully functional local chatbot interface built with Hugging Face's text generation models. This project demonstrates integration of language models into a CLI environment with conversation memory management and factual response capabilities.

## 🎯 Features

- **Local Execution**: Runs entirely on your machine (GPU optional)
- **Conversation Memory**: Maintains last 5 turns using sliding window
- **Factual Responses**: Built-in knowledge base for capitals and tourist attractions
- **Context Awareness**: Understands follow-up questions and references
- **Modular Design**: Clean separation of concerns across modules
- **Graceful Exit**: Type `/exit` to quit smoothly
- **Command Support**: Built-in commands for better UX

## 📋 Requirements

- Python 3.7+
- pip package manager
- Jupyter (optional, for running the prototype notebook)

## 🚀 Setup Instructions

### 1. Clone or Download the Project

```bash
git clone <your-repo-url>
cd chatbot_project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `transformers` - Hugging Face library
- `torch` - PyTorch for model execution

### 3. Run the Chatbot

```bash
python interface.py
```

## 💬 Usage

### Starting a Conversation

Once the chatbot starts, you can type naturally:

```
User: What is the capital of France?
Bot: The capital of France is Paris.

User: And what about Italy?
Bot: The capital of Italy is Rome.
```

### Available Commands

- `/exit` - Exit the chatbot
- `/clear` - Clear conversation history
- `/help` - Display help message

### Sample Interaction

```
User: What is the capital of France?
Bot: The capital of France is Paris.

User: Tell me about places to visit there
Bot: France offers many famous attractions including the iconic Eiffel Tower, 
    the Louvre Museum (home to the Mona Lisa), the Palace of Versailles, 
    Mont Saint-Michel, and the beautiful French Riviera.

User: What about Italy?
Bot: The capital of Italy is Rome.
```

## 🧪 Jupyter Prototype

The project includes a Jupyter notebook (`chatbot_prototype.ipynb`) for interactive testing and development:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `chatbot_prototype.ipynb`

The notebook demonstrates:
- Model initialization and setup
- Conversation flow testing
- Capital and places query handling
- Context preservation
- Unit tests for core functionality

## 🏗️ Project Structure

```
local-command-line-chatbot/
│
├── model_loader.py         # Model loading and response generation
├── chat_memory.py         # Conversation memory buffer
├── interface.py           # CLI interface
├── chatbot_prototype.ipynb # Interactive testing notebook
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

### Module Descriptions

**model_loader.py**
- Loads Hugging Face model (default: distilgpt2)
- Manages factual response database
- Handles context preservation and query parsing
- Generates responses with customizable parameters

**chat_memory.py**
- Implements sliding window buffer for conversation history
- Maintains last N turns (configurable, default: 5)
- Tracks query types for better context handling

**interface.py**
- Provides CLI interface for user interaction
- Integrates model and memory components
- Handles commands and graceful exit

## ⚙️ Configuration

You can customize the chatbot by modifying parameters in `interface.py`:

```python
chatbot = ChatbotInterface(
    model_name="distilgpt2",  # Change to any HF model
    memory_turns=5            # Adjust memory window size
)
```

### Knowledge Base

The chatbot includes built-in knowledge for:
- Capital cities of major countries
- Tourist attractions and places to visit
- More categories can be added in `model_loader.py`

## 📝 Technical Details

### Memory Management
Uses a sliding window approach with `collections.deque` to maintain the most recent conversation turns. This ensures:
- Efficient memory usage
- Relevant context for coherent responses
- Automatic cleanup of old messages

### Model Generation
Text generation configured with:
- Temperature: 0.7 (balanced creativity)
- Top-p sampling: 0.9 (nucleus sampling)
- Dynamic max length based on context

## 🐛 Troubleshooting

**Issue**: Model download is slow
- **Solution**: First run downloads model (~250MB for distilgpt2). Subsequent runs use cached model.

**Issue**: Out of memory errors
- **Solution**: Use a smaller model like `distilgpt2` or reduce `memory_turns`

**Issue**: Context not preserved
- **Solution**: Ensure queries are related and within the memory window (5 turns)

## 🚀 Coming Soon

- Command-line options for model selection
- Expanded knowledge base
- Enhanced evaluation framework

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Raghava Ram
[GitHub Profile](https://github.com/Raghava-Ram)