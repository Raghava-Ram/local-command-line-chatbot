"""
interface.py
Command-line interface for the chatbot
"""

from model_loader import ModelLoader
from chat_memory import ChatMemory
import sys

class ChatbotInterface:
    def __init__(self, model_name="distilgpt2", memory_turns=5):
        """
        Initialize the chatbot interface.
        
        Args:
            model_name (str): Hugging Face model name
            memory_turns (int): Number of conversation turns to remember
        """
        self.model_loader = ModelLoader(model_name)
        self.memory = ChatMemory(max_turns=memory_turns)
        self.is_running = False
        
    def initialize(self):
        """Load the model and prepare the chatbot."""
        print("=" * 60)
        print("  LOCAL CHATBOT - Hugging Face CLI Interface")
        print("=" * 60)
        print()
        
        try:
            self.model_loader.load_model()
            print()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def display_help(self):
        """Display available commands."""
        print("\nAvailable commands:")
        print("  /exit    - Exit the chatbot")
        print("  /clear   - Clear conversation history")
        print("  /help    - Show this help message")
        print()
    
    def run(self):
        """Start the chatbot interaction loop."""
        if not self.initialize():
            return
        
        self.display_help()
        print("Start chatting! (Type /exit to quit)\n")
        
        self.is_running = True
        
        while self.is_running:
            try:
                try:
                    # Get user input
                    user_input = input("User: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n\nExiting chatbot. Goodbye!")
                    break
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    self.handle_command(user_input)
                    continue
                
                # Update memory before generating response
                topic, context, initial_type = self.model_loader.parse_query(user_input)
                self.memory.add_message("User", user_input, initial_type)
                
                try:
                    # Get bot response with updated history
                    response_tuple = self.model_loader.generate_response(
                        user_input,  # Pass just the user input
                        conversation_history=self.memory.buffer,  # Pass the updated history
                        max_new_tokens=50  # Allow slightly longer responses
                    )
                    
                    # Unpack response and query type
                    if isinstance(response_tuple, tuple):
                        bot_response, query_type = response_tuple
                    else:
                        bot_response, query_type = response_tuple, None
                    
                    # Clean up response
                    bot_response = self.clean_response(bot_response)
                    
                    # Validate response
                    if self.is_valid_response(bot_response, user_input):
                        # Update memory with the bot response and query type
                        self.memory.add_message("Bot", bot_response, query_type)
                        
                        # Display response
                        print(f"Bot: {bot_response}\n")
                    else:
                        # Generate a fallback response
                        fallback = self.generate_fallback_response(user_input)
                        
                        # Update memory with fallback response
                        self.memory.add_message("Bot", fallback, query_type)
                        
                        print(f"Bot: {fallback}\n")
                    
                except KeyboardInterrupt:
                    print("\n\nExiting chatbot. Goodbye!")
                    break
                except Exception as e:
                    print(f"Error generating response: {str(e)}")
                    print("Bot: I apologize, but I'm having trouble generating a response right now.\n")
                    continue
                
            except Exception as e:
                print(f"\nAn unexpected error occurred: {str(e)}")
                print("Please try again or type /exit to quit.\n")
                continue
    
    def clean_response(self, response):
        """
        Clean and format the bot response.
        
        Args:
            response (str): Raw model output
            
        Returns:
            str: Cleaned response
        """
        import re

        # Check if response is empty or None
        if not response:
            return "I apologize, but I'm having trouble generating a response."
        
        # Remove common unwanted patterns
        unwanted_patterns = [
            r'\[.*?\]',          # Text in square brackets
            r'Bot:',             # Speaker labels
            r'User:',
            r'Human:',
            r'Assistant:',
            r'System:',
            r'Instructions:.*?\\n\\n',  # Remove instruction block
            r'Example interactions:.*?Current conversation:',  # Remove examples
            r'This person would be',     # Common irrelevant starts
            r'No other word for',
            r'Or do anyone'
        ]
        
        for pattern in unwanted_patterns:
            response = re.sub(pattern, '', response)
        
        # Clean up the text
        response = ' '.join(response.split())  # Normalize whitespace
        
        # Split into sentences and get coherent ones
        sentences = []
        for sent in re.split(r'[.!?]+', response):
            sent = sent.strip()
            if sent and len(sent.split()) > 2:  # Only keep sentences with 3+ words
                sentences.append(sent)
        
        # Take up to 2 coherent sentences
        if sentences:
            response = '. '.join(sentences[:2]) + '.'
        else:
            response = "I apologize, but I need more context to provide a meaningful response."
        
        return response
        
        return response
    
    def handle_command(self, command):
        """
        Handle special commands.
        
        Args:
            command (str): The command to execute
        """
        command = command.lower()
        
        if command == "/exit":
            self.exit_chatbot()
        elif command == "/clear":
            self.memory.clear()
            print("Conversation history cleared.\n")
        elif command == "/help":
            self.display_help()
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands.\n")
    
    def is_valid_response(self, response, user_input):
        """
        Validate if the response is appropriate for the input.
        
        Args:
            response (str): The generated response
            user_input (str): The user's input
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        # Check for minimum length
        if len(response.split()) < 3:
            return False
            
        # Check if response is just repeating the input
        if user_input.lower() in response.lower():
            return False
            
        # Check for common meaningless patterns
        meaningless_patterns = [
            r'would be the',
            r'this person',
            r'no other word',
            r'or do anyone',
            r'use your email',
            r'social media accounts',
            r'let\'s start by',
            r'looking at how',
            r'have to talk about',
            r'speaking (?:french|spanish|english)',
            r'interaction has taken',
            r'need .* assistants',
            r'someone named',
            r'different countries'
        ]
        
        import re
        for pattern in meaningless_patterns:
            if re.search(pattern, response.lower()):
                return False
                
        # Check for question marks in response to questions
        if any(w in user_input.lower() for w in ['what', 'where', 'when', 'why', 'how']):
            if '?' in response:
                return False
                
        # Response shouldn't ask a question when greeting
        if any(greeting in user_input.lower() for greeting in ['hi', 'hello', 'hey']):
            if '?' in response:
                return False
        
        return True
        
    def generate_fallback_response(self, user_input):
        """
        Generate a fallback response when the model's response is invalid.
        
        Args:
            user_input (str): The user's input
            
        Returns:
            str: A contextual fallback response
        """
        import random
        
        # Greetings responses
        greetings_responses = [
            "Hello! How can I help you today?",
            "Hi there! What can I assist you with?",
            "Greetings! How may I help you?",
            "Hello! Feel free to ask me anything."
        ]
        
        # Question fallbacks
        question_fallbacks = [
            "I apologize, but I need more information to answer that question accurately.",
            "That's an interesting question. Could you provide more details?",
            "I want to give you an accurate answer. Could you be more specific?",
            "I'm not entirely sure about that. Could you rephrase your question?"
        ]
        
        # General fallbacks
        general_fallbacks = [
            "I understand, but could you rephrase that? I want to make sure I give you a helpful response.",
            "I'm not sure I fully understood. Could you explain in a different way?",
            "Could you provide more context? That would help me give a better response.",
            "I want to help, but I need a bit more clarity. Could you elaborate?"
        ]
        
        # Check for common question patterns
        if any(q in user_input.lower() for q in ['what', 'where', 'when', 'why', 'how']):
            return random.choice(question_fallbacks)
        elif any(greeting in user_input.lower() for greeting in ['hi', 'hello', 'hey']):
            return random.choice(greetings_responses)
        else:
            return random.choice(general_fallbacks)
    
    def exit_chatbot(self):
        """Exit the chatbot gracefully."""
        print("\nExiting chatbot. Goodbye!")
        self.is_running = False
        sys.exit(0)

def main():
    """Main entry point for the chatbot application."""
    chatbot = ChatbotInterface(
        model_name="distilgpt2",  # Small, fast model
        memory_turns=5
    )
    chatbot.run()

if __name__ == "__main__":
    main()