"""
chat_memory.py
Manages conversation history using a sliding window buffer
"""

from collections import deque

class ChatMemory:
    def __init__(self, max_turns=5):
        """
        Initialize the chat memory with a sliding window.
        
        Args:
            max_turns (int): Maximum number of conversation turns to remember
        """
        self.max_turns = max_turns
        self.buffer = deque(maxlen=max_turns * 2)  # *2 for user + bot messages
        
    def add_message(self, role, message, query_type=None):
        """
        Add a message to the conversation buffer.
        
        Args:
            role (str): Either 'User' or 'Bot'
            message (str): The message content
            query_type (str): Type of query (capital, places, etc.)
        """
        self.buffer.append({
            "role": role, 
            "message": message,
            "query_type": query_type
        })
    
    def get_context(self):
        """
        Get the conversation context as a formatted string.
        
        Returns:
            str: Formatted conversation history
        """
        if not self.buffer:
            return ""
        
        context = "\n".join([
            f"{entry['role']}: {entry['message']}"
            for entry in self.buffer
        ])
        
        return context
    
    def get_prompt(self, new_user_input):
        """
        Create a prompt combining conversation history and new input.
        
        Args:
            new_user_input (str): The latest user input
            
        Returns:
            str: Complete prompt for the model
        """
        # Start with basic instructions for better responses
        base_prompt = (
            "The following is a conversation with an AI assistant. "
            "The assistant is helpful, knowledgeable, and direct.\n\n"
        )
        
        context = self.get_context()
        
        if context:
            # Format each turn of conversation clearly
            prompt = f"{base_prompt}{context}\nUser: {new_user_input}\nAssistant:"
        else:
            prompt = f"{base_prompt}User: {new_user_input}\nAssistant:"
        
        return prompt
    
    def clear(self):
        """Clear all conversation history."""
        self.buffer.clear()
    
    def get_history_length(self):
        """Get the current number of messages in buffer."""
        return len(self.buffer)