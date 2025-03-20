import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import json
from ..rag.retriever import MultiLanguageRetriever
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EducationalTutor:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        supported_languages: List[str] = ["en"],
        max_retries: int = 3
    ):
        self.device = torch.device("cpu")  # Force CPU usage
        self.max_retries = max_retries
        
        try:
            # Load model and tokenizer
            logger.info(f"Loading model {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None  # No device mapping for CPU
            )
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Initialize RAG
        try:
            self.retriever = MultiLanguageRetriever(supported_languages)
            logger.info("RAG initialized successfully")
            
            # Load curriculum data
            curriculum_dir = os.getenv("CURRICULUM_DIR", "data/curriculum")
            for lang in supported_languages:
                self.retriever.load_curriculum(curriculum_dir, lang)
            logger.info("Curriculum data loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            raise
    
    def _format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format the prompt with context and question"""
        # Format for TinyLlama chat model
        prompt = "<|system|>You are an educational tutor. Your role is to guide students through problems step by step, encouraging them to think independently.</s>\n"
        if context:
            prompt += f"<|context|>{context}</s>\n"
        prompt += f"<|user|>{question}</s>\n<|assistant|>"
        return prompt
    
    def _generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate response from the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the response after the assistant token
        response = response.split("<|assistant|>")[-1].strip()
        return response
    
    def _get_relevant_context(self, question: str, language: str = "en") -> str:
        """Get relevant context from curriculum"""
        results = self.retriever.retrieve_relevant_content(question, language)
        return "\n".join([r["content"] for r in results])
    
    def tutor(
        self,
        question: str,
        language: str = "en",
        difficulty_level: str = "medium"
    ) -> Dict:
        """Main tutoring function"""
        # Get relevant context
        context = self._get_relevant_context(question, language)
        
        # Format initial prompt
        prompt = self._format_prompt(question, context)
        
        # Generate initial response
        response = self._generate_response(prompt)
        
        # Check if response needs improvement
        retry_count = 0
        while retry_count < self.max_retries:
            if self._needs_improvement(response):
                # Add more specific guidance
                prompt += f"<hint>Let me guide you more specifically.</hint>\n"
                response = self._generate_response(prompt)
                retry_count += 1
            else:
                break
        
        return {
            "question": question,
            "context": context,
            "response": response,
            "retries": retry_count,
            "language": language,
            "difficulty_level": difficulty_level
        }
    
    def _needs_improvement(self, response: str) -> bool:
        """Check if the response needs improvement"""
        # Add your criteria for response quality
        # For example, check for:
        # 1. Response length
        # 2. Presence of key concepts
        # 3. Clarity of explanation
        # 4. Step-by-step structure
        return len(response.split()) < 50 or "step" not in response.lower()
    
    def save_session(self, session_data: Dict, output_path: str):
        """Save tutoring session data"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    def load_session(self, session_path: str) -> Dict:
        """Load tutoring session data"""
        with open(session_path, "r", encoding="utf-8") as f:
            return json.load(f)

def main():
    # Example usage
    tutor = EducationalTutor(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use TinyLlama model
        supported_languages=["en", "es"]
    )
    
    # Example question
    question = "How do I solve quadratic equations?"
    
    # Get tutoring response
    response = tutor.tutor(question)
    
    # Save session
    os.makedirs("sessions", exist_ok=True)  # Create sessions directory if it doesn't exist
    tutor.save_session(response, "sessions/example_session.json")
    
    print("Question:", response["question"])
    print("\nResponse:", response["response"])

if __name__ == "__main__":
    main() 