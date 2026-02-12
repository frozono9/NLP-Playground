#!/usr/bin/env bash
# exit on error
set -o errexit

python --version
pip install --upgrade pip

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('wordnet'); nltk.download('sentiwordnet'); nltk.download('omw-1.4')"

# Download spaCy model
python -m spacy download en_core_web_sm
