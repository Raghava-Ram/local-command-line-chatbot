from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import traceback
import random

class ModelLoader:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.generator = None
        self.tokenizer = None
        self.current_context = None
        self.factual_responses = {
            "capital_responses": {
                "france": "The capital of France is Paris.",
                "india": "The capital of India is New Delhi.",
                "italy": "The capital of Italy is Rome.",
                "united states": "The capital of United States is Washington, D.C.",
                "usa": "The capital of United States is Washington, D.C.",
                "united kingdom": "The capital of United Kingdom is London.",
                "uk": "The capital of United Kingdom is London.",
                "germany": "The capital of Germany is Berlin.",
                "spain": "The capital of Spain is Madrid.",
                "canada": "The capital of Canada is Ottawa.",
                "australia": "The capital of Australia is Canberra.",
                "japan": "The capital of Japan is Tokyo.",
                "china": "The capital of China is Beijing.",
                "russia": "The capital of Russia is Moscow.",
                "brazil": "The capital of Brazil is Brasilia.",
                "mexico": "The capital of Mexico is Mexico City."
            },
            "place_responses": {
                "france": "France offers many famous attractions including the iconic Eiffel Tower, the Louvre Museum (home to the Mona Lisa), the Palace of Versailles, Mont Saint-Michel, and the beautiful French Riviera.",
                "paris": "Paris has many famous attractions including the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, Arc de Triomphe, Champs-Elysees, and Montmartre.",
                "italy": "Italy offers many famous attractions including the Colosseum and Roman Forum in Rome, the canals of Venice, the Leaning Tower of Pisa, Florence's Renaissance art and architecture, and the beautiful Amalfi Coast.",
                "rome": "Rome has many famous attractions including the Colosseum, Roman Forum, Vatican City (with St. Peter's Basilica), the Pantheon, Trevi Fountain, Spanish Steps, and countless museums and piazzas.",
                "india": "India has many interesting places including the Taj Mahal in Agra, the historic Red Fort in Delhi, the sacred city of Varanasi, the beaches of Goa, and the backwaters of Kerala."
            },
            "greetings": [
                "Hello! How can I help you today?",
                "Hi there! What can I assist you with?",
                "Greetings! How may I help you?"
            ]
        }

    def load_model(self):
        print(f"Loading model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            if self.device == -1:
                print("Device set to use CPU")
            print("Model loaded successfully!")
            return self.generator
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            raise

    def parse_query(self, user_input, conversation_history=None):
        try:
            user_input = user_input.lower().strip()
            places = list(self.factual_responses["capital_responses"].keys()) + ["paris", "rome"]
            
            if any(greeting in user_input for greeting in ["hi", "hello", "hey"]):
                return None, None, "greeting"
            
            followup_words = ["what about", "how about", "what of", "and", "what is"]
            there_words = ["there", "that place", "that country", "it"]
            is_followup = any(word in user_input for word in followup_words)
            is_there_reference = any(word in user_input for word in there_words)
            
            mentioned_place = next((place for place in places if place in user_input), None)
            
            if is_there_reference and self.current_context and not mentioned_place:
                mentioned_place = self.current_context
                mentioned_place = self.current_context
            
            is_capital = any(word in user_input for word in ["capital", "capitol"])
            is_places = any(word in user_input for word in [
                "visit", "places", "attractions", "see", "where", 
                "tell me about", "things", "tourists"
            ])
            
            if (is_followup or is_there_reference) and conversation_history:
                last_query_type = None
                last_place = mentioned_place or self.current_context
                
                for entry in reversed(conversation_history):
                    if isinstance(entry, dict) and entry.get("message"):
                        msg = entry["message"].lower()
                        entry_type = entry.get("query_type")
                        
                        # Only set the last_query_type if we don't have an explicit type
                        if not (is_capital or is_places):
                            if entry_type == "capital":
                                last_query_type = "capital"
                                if last_place:  # If we have a place, we can stop
                                    break
                            elif not last_query_type and entry_type:
                                last_query_type = entry_type
                            
                        if not last_place:
                            for place in places:
                                if place in msg.lower():
                                    last_place = place
                                    self.current_context = place
                                    if entry_type == "capital":  # Perfect match!
                                        break
                                        
                if last_place:
                    self.current_context = last_place
                    
                    # Use current query type if explicitly mentioned
                    if is_capital:
                        return last_place, last_place, "capital"
                    elif is_places:
                        return last_place, last_place, "places"
                    elif last_query_type:  # Fallback to historical type
                        return last_place, last_place, last_query_type
            
            if mentioned_place:
                self.current_context = mentioned_place
                if is_capital:
                    return mentioned_place, mentioned_place, "capital"
                elif is_places:
                    return mentioned_place, mentioned_place, "places"
                else:
                    if conversation_history:
                        for entry in reversed(conversation_history):
                            if isinstance(entry, dict):
                                if entry.get("query_type") == "capital":
                                    return mentioned_place, mentioned_place, "capital"
                                elif entry.get("query_type") == "places":
                                    return mentioned_place, mentioned_place, "places"
                    return mentioned_place, mentioned_place, "places"
            
            return None, None, "general"
            
        except Exception as e:
            print(f"Error parsing query: {str(e)}")
            print(traceback.format_exc())
            return None, None, "general"

    def generate_response(self, prompt, conversation_history=None, max_new_tokens=50):
        try:
            if not prompt or not isinstance(prompt, str):
                return "I'm sorry, I didn't receive a valid question. Could you please rephrase?", "error"

            topic, context, query_type = self.parse_query(prompt, conversation_history)
            
            if query_type == "greeting":
                return random.choice(self.factual_responses["greetings"]), query_type
            
            if query_type == "capital" and topic:
                if topic in self.factual_responses["capital_responses"]:
                    return self.factual_responses["capital_responses"][topic], query_type
            
            if query_type == "places" and topic:
                if topic in self.factual_responses["place_responses"]:
                    return self.factual_responses["place_responses"][topic], query_type
            
            if self.generator is None:
                return "I apologize, but I need the model to be loaded first. Please try again.", "error"

            try:
                response = self.generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.85,
                    top_k=20,
                    repetition_penalty=1.5,
                    truncation=True,
                    clean_up_tokenization_spaces=True,
                    no_repeat_ngram_size=3
                )
                
                generated_text = response[0]["generated_text"]
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                final_response = generated_text or "I'm not sure how to answer that. Could you please rephrase?"
                return final_response, query_type
                
            except Exception as e:
                print(f"Error during text generation: {str(e)}")
                print(traceback.format_exc())
                return "I apologize, but I'm having trouble generating a response right now.", "error"
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            print(traceback.format_exc())
            return "I apologize, but I encountered an error. Could you try asking again?", "error"
