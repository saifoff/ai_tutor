import os
import json
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from .tutor.tutor import EducationalTutor
from dotenv import load_dotenv

load_dotenv()

class TutorEvaluator:
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        supported_languages: List[str] = ["en"]
    ):
        self.tutor = EducationalTutor(
            model_path=model_path,
            supported_languages=supported_languages
        )
        self.test_data = self._load_test_data(test_data_path)
    
    def _load_test_data(self, test_data_path: str) -> List[Dict]:
        """Load test data from JSON file"""
        with open(test_data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate(self) -> Dict:
        """Evaluate the tutor's performance"""
        results = {
            "total_questions": len(self.test_data),
            "correct_answers": 0,
            "average_response_length": 0,
            "average_retries": 0,
            "language_performance": {},
            "difficulty_performance": {},
            "response_times": []
        }
        
        for test_case in tqdm(self.test_data, desc="Evaluating"):
            # Get tutor's response
            response = self.tutor.tutor(
                question=test_case["question"],
                language=test_case.get("language", "en"),
                difficulty_level=test_case.get("difficulty", "medium")
            )
            
            # Update metrics
            results["correct_answers"] += self._check_answer(
                response["response"],
                test_case["expected_answer"]
            )
            
            results["average_response_length"] += len(response["response"].split())
            results["average_retries"] += response["retries"]
            results["response_times"].append(response.get("response_time", 0))
            
            # Update language-specific metrics
            lang = test_case.get("language", "en")
            if lang not in results["language_performance"]:
                results["language_performance"][lang] = {
                    "total": 0,
                    "correct": 0
                }
            results["language_performance"][lang]["total"] += 1
            results["language_performance"][lang]["correct"] += self._check_answer(
                response["response"],
                test_case["expected_answer"]
            )
            
            # Update difficulty-specific metrics
            diff = test_case.get("difficulty", "medium")
            if diff not in results["difficulty_performance"]:
                results["difficulty_performance"][diff] = {
                    "total": 0,
                    "correct": 0
                }
            results["difficulty_performance"][diff]["total"] += 1
            results["difficulty_performance"][diff]["correct"] += self._check_answer(
                response["response"],
                test_case["expected_answer"]
            )
        
        # Calculate final metrics
        results["accuracy"] = results["correct_answers"] / results["total_questions"]
        results["average_response_length"] /= results["total_questions"]
        results["average_retries"] /= results["total_questions"]
        results["average_response_time"] = np.mean(results["response_times"])
        
        # Calculate language-specific accuracies
        for lang in results["language_performance"]:
            results["language_performance"][lang]["accuracy"] = (
                results["language_performance"][lang]["correct"] /
                results["language_performance"][lang]["total"]
            )
        
        # Calculate difficulty-specific accuracies
        for diff in results["difficulty_performance"]:
            results["difficulty_performance"][diff]["accuracy"] = (
                results["difficulty_performance"][diff]["correct"] /
                results["difficulty_performance"][diff]["total"]
            )
        
        return results
    
    def _check_answer(self, response: str, expected: str) -> bool:
        """Check if the response matches the expected answer"""
        # Implement your answer checking logic here
        # This could involve:
        # 1. Exact matching
        # 2. Semantic similarity
        # 3. Key concept extraction and matching
        # 4. Multiple choice answer selection
        return True  # Placeholder
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    # Initialize evaluator
    evaluator = TutorEvaluator(
        model_path=os.getenv("MODEL_PATH", "models/educational_tutor"),
        test_data_path="data/evaluation/test_data.json",
        supported_languages=["en", "es"]
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Save results
    evaluator.save_results(results, "evaluation/results.json")
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Average Response Length: {results['average_response_length']:.1f} words")
    print(f"Average Retries: {results['average_retries']:.1f}")
    print(f"Average Response Time: {results['average_response_time']:.2f} seconds")
    
    print("\nLanguage-specific Performance:")
    for lang, metrics in results["language_performance"].items():
        print(f"{lang}: {metrics['accuracy']:.2%}")
    
    print("\nDifficulty-specific Performance:")
    for diff, metrics in results["difficulty_performance"].items():
        print(f"{diff}: {metrics['accuracy']:.2%}")

if __name__ == "__main__":
    main() 