from setuptools import setup, find_packages

setup(
    name="ai_tutor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "sentencepiece>=0.1.99",
        "faiss-cpu>=1.7.4",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.10",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "wandb>=0.15.12"
    ],
) 