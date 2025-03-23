# AI Educational Tutor

An intelligent tutoring system powered by Mistral-7B that provides personalized educational support with RAG capabilities.

## Features

1. Personalized Educational Tutoring
   - Step-by-step guided learning
   - Socratic teaching approach
   - Adaptive difficulty levels
   - Clear explanations of underlying principles

2. RAG with School Curriculum
   - Integration with standard school textbooks
   - Multi-language support
   - Context-aware responses
   - Curriculum-aligned content

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with:
```
WANDB_API_KEY=your_wandb_key
MODEL_PATH=path_to_save_model
```

3. Prepare your curriculum data:
Place your curriculum documents in the `data/curriculum` directory.

## Project Structure

- `src/`
  - `model/`: Model training and fine-tuning code
  - `rag/`: RAG implementation
  - `tutor/`: Tutoring logic and interaction handling
  - `utils/`: Utility functions
- `data/`
  - `curriculum/`: Curriculum documents
  - `training/`: Training data
- `configs/`: Configuration files
- `evaluation/`: Model evaluation scripts

## Usage

1. Train the model:
```bash
python src/train.py
```

2. Run the tutor:
```bash
python src/tutor.py
```

## Evaluation

Run evaluation scripts:
```bash
python src/evaluate.py
```

## License

MIT License 
