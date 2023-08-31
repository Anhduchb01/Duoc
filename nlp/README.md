# Install Pip
python3 -m pip install -U pip
python3 -m pip install -U setuptools

# Install requirements
pip install -r requirements.txt

# Install NLTK packages
nltk.download()

# Install Spacy
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download ja_core_news_sm

# Start app
python app.py
