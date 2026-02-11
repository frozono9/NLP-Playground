from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import nltk
import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Download required NLTK data
required_nltk_data = [
    ('tokenizers/punkt', 'punkt'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
    ('corpora/words', 'words'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/sentiwordnet', 'sentiwordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
]

for path, name in required_nltk_data:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize stemmers and lemmatizer
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# Code templates for educational purposes
CODE_TEMPLATES = {
    'nltk_tokenize': '''import nltk

text = "{text}"
tokens = nltk.word_tokenize(text)
print(f"Tokens: {{tokens}}")
print(f"Count: {{len(tokens)}}")''',
    
    'nltk_pos_tag': '''import nltk

text = "{text}"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

for word, tag in pos_tags:
    print(f"{{word:15}} -> {{tag}}")''',
    
    'nltk_ner': '''import nltk

text = "{text}"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
chunks = nltk.ne_chunk(pos_tags)

for chunk in chunks:
    if hasattr(chunk, 'label'):
        entity = ' '.join(c[0] for c in chunk)
        label = chunk.label()
        print(f"{{entity}} -> {{label}}")''',
    
    'spacy_analyze': '''import spacy

nlp = spacy.load('en_core_web_sm')
text = "{text}"
doc = nlp(text)

# Tokens
print("Tokens:", [token.text for token in doc])

# POS Tags
for token in doc:
    print(f"{{token.text:15}} -> {{token.tag_}} ({{token.pos_}})")

# Entities
for ent in doc.ents:
    print(f"{{ent.text}} -> {{ent.label_}}")''',
    
    'spacy_dependency': '''import spacy

nlp = spacy.load('en_core_web_sm')
text = "{text}"
doc = nlp(text)

for token in doc:
    print(f"{{token.text:15}} -> {{token.dep_:10}} -> {{token.head.text}}")''',
    
    'wordnet_synonyms': '''from nltk.corpus import wordnet as wn

word = "{word}"
synsets = wn.synsets(word)

for syn in synsets:
    print(f"Synset: {{syn.name()}}")
    print(f"  Definition: {{syn.definition()}}")
    print(f"  Lemmas: {{[l.name() for l in syn.lemmas()]}}")
    print()''',
    
    'wordnet_hypernyms': '''from nltk.corpus import wordnet as wn

word = "{word}"
synsets = wn.synsets(word)

if synsets:
    syn = synsets[0]
    hypernyms = syn.hypernyms()
    
    print(f"Word: {{word}}")
    print(f"Hypernyms (broader terms):")
    for hyp in hypernyms:
        print(f"  - {{hyp.name()}}: {{hyp.definition()}}")''',
    
    'wordnet_similarity': '''from nltk.corpus import wordnet as wn

word1 = "{word1}"
word2 = "{word2}"

synsets1 = wn.synsets(word1)
synsets2 = wn.synsets(word2)

if synsets1 and synsets2:
    similarity = synsets1[0].path_similarity(synsets2[0])
    print(f"Similarity between '{{word1}}' and '{{word2}}': {{similarity}}")''',
    
    'sentiment': '''from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import nltk

text = "{text}"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

total_pos = 0
total_neg = 0
count = 0

for word, tag in pos_tags:
    synsets = wn.synsets(word)
    if synsets:
        swn_synset = swn.senti_synset(synsets[0].name())
        total_pos += swn_synset.pos_score()
        total_neg += swn_synset.neg_score()
        count += 1

if count > 0:
    avg_pos = total_pos / count
    avg_neg = total_neg / count
    print(f"Positive: {{avg_pos:.3f}}")
    print(f"Negative: {{avg_neg:.3f}}")
    print(f"Neutral: {{1 - avg_pos - avg_neg:.3f}}")''',
    
    'jaccard_similarity': '''def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

text1 = "{text1}"
text2 = "{text2}"

similarity = jaccard_similarity(text1, text2)
print(f"Jaccard Similarity: {{similarity:.3f}}")''',
    
    'cosine_similarity': '''from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text1 = "{text1}"
text2 = "{text2}"

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform([text1, text2])
similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

print(f"Cosine Similarity: {{similarity:.3f}}")''',
    
    'stemming': '''from nltk.stem import PorterStemmer, LancasterStemmer

text = "{text}"
words = text.split()

porter = PorterStemmer()
lancaster = LancasterStemmer()

print("Porter Stemmer:")
for word in words:
    print(f"  {{word:15}} -> {{porter.stem(word)}}")

print("\\nLancaster Stemmer:")
for word in words:
    print(f"  {{word:15}} -> {{lancaster.stem(word)}}")''',
    
    'lemmatization': '''from nltk.stem import WordNetLemmatizer
import nltk

text = "{text}"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

lemmatizer = WordNetLemmatizer()

for word, tag in pos_tags:
    lemma = lemmatizer.lemmatize(word.lower())
    print(f"{{word:15}} -> {{lemma}}")''',
    
    'ngrams': '''from nltk import ngrams
import nltk

text = "{text}"
tokens = nltk.word_tokenize(text)
n = {n}

n_grams = list(ngrams(tokens, n))

print(f"{{n}}-grams:")
for gram in n_grams:
    print(f"  {{' '.join(gram)}}")'''
}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/code/<task>', methods=['POST'])
def get_code(task):
    data = request.json
    template = CODE_TEMPLATES.get(task, '')
    
    # Format template with provided data
    try:
        code = template.format(**data)
    except KeyError:
        code = template
    
    return jsonify({'code': code})

# Existing endpoints
@app.route('/api/nltk/tokenize', methods=['POST'])
def nltk_tokenize():
    data = request.json
    text = data.get('text', '')
    tokens = nltk.word_tokenize(text)
    return jsonify({'tokens': tokens, 'count': len(tokens)})

@app.route('/api/nltk/pos_tag', methods=['POST'])
def nltk_pos_tag():
    data = request.json
    text = data.get('text', '')
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return jsonify({'pos_tags': [{'word': word, 'tag': tag} for word, tag in pos_tags]})

@app.route('/api/nltk/ner', methods=['POST'])
def nltk_ner():
    data = request.json
    text = data.get('text', '')
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(pos_tags)
    
    entities = []
    for item in chunks:
        if hasattr(item, 'label'):
            entities.append({
                'text': ' '.join(word for word, tag in item),
                'label': item.label()
            })
    
    return jsonify({'entities': entities})

@app.route('/api/spacy/analyze', methods=['POST'])
def spacy_analyze():
    if nlp is None:
        return jsonify({'error': 'spaCy model not loaded'}), 500
    
    data = request.json
    text = data.get('text', '')
    doc = nlp(text)
    
    return jsonify({
        'tokens': [token.text for token in doc],
        'pos_tags': [{'word': token.text, 'tag': token.tag_, 'pos': token.pos_} for token in doc],
        'entities': [{'text': ent.text, 'label': ent.label_} for ent in doc.ents],
        'dependencies': [{
            'word': token.text,
            'tag': token.tag_,
            'dep': token.dep_,
            'head': token.head.text
        } for token in doc]
    })

@app.route('/api/spacy/dependency_tree', methods=['POST'])
def spacy_dependency_tree():
    if nlp is None:
        return jsonify({'error': 'spaCy model not loaded'}), 500
    
    data = request.json
    text = data.get('text', '')
    doc = nlp(text)
    
    nodes = [{'id': token.i, 'text': token.text, 'tag': token.tag_, 'dep': token.dep_} for token in doc]
    edges = [{'source': token.head.i, 'target': token.i, 'label': token.dep_} 
             for token in doc if token.head.i != token.i]
    
    return jsonify({'nodes': nodes, 'edges': edges})

# New WordNet endpoints
@app.route('/api/wordnet/synonyms', methods=['POST'])
def wordnet_synonyms():
    data = request.json
    word = data.get('word', '').lower()
    
    synsets = wn.synsets(word)
    results = []
    
    for syn in synsets:
        results.append({
            'synset': syn.name(),
            'definition': syn.definition(),
            'lemmas': [l.name() for l in syn.lemmas()],
            'examples': syn.examples()
        })
    
    return jsonify({'word': word, 'synsets': results})

@app.route('/api/wordnet/hypernyms', methods=['POST'])
def wordnet_hypernyms():
    data = request.json
    word = data.get('word', '').lower()
    
    synsets = wn.synsets(word)
    if not synsets:
        return jsonify({'word': word, 'hypernyms': []})
    
    syn = synsets[0]
    hypernyms = []
    
    for hyp in syn.hypernyms():
        hypernyms.append({
            'name': hyp.name(),
            'definition': hyp.definition()
        })
    
    return jsonify({'word': word, 'synset': syn.name(), 'hypernyms': hypernyms})

@app.route('/api/wordnet/similarity', methods=['POST'])
def wordnet_similarity():
    data = request.json
    word1 = data.get('word1', '').lower()
    word2 = data.get('word2', '').lower()
    
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    
    if not synsets1 or not synsets2:
        return jsonify({'similarity': None, 'error': 'One or both words not found'})
    
    similarity = synsets1[0].path_similarity(synsets2[0])
    
    return jsonify({
        'word1': word1,
        'word2': word2,
        'similarity': similarity
    })

# Sentiment Analysis
@app.route('/api/sentiment/analyze', methods=['POST'])
def sentiment_analyze():
    data = request.json
    text = data.get('text', '')
    
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    total_pos = 0
    total_neg = 0
    count = 0
    word_scores = []
    
    for word, tag in pos_tags:
        synsets = wn.synsets(word.lower())
        if synsets:
            swn_synset = swn.senti_synset(synsets[0].name())
            if swn_synset.pos_score() > 0 or swn_synset.neg_score() > 0:
                total_pos += swn_synset.pos_score()
                total_neg += swn_synset.neg_score()
                count += 1
                word_scores.append({
                    'word': word,
                    'pos': round(swn_synset.pos_score(), 3),
                    'neg': round(swn_synset.neg_score(), 3),
                    'obj': round(swn_synset.obj_score(), 3)
                })
    
    if count > 0:
        avg_pos = total_pos / count
        avg_neg = total_neg / count
        avg_obj = 1 - avg_pos - avg_neg
    else:
        avg_pos = avg_neg = avg_obj = 0
    
    return jsonify({
        'overall': {
            'positive': round(avg_pos, 3),
            'negative': round(avg_neg, 3),
            'objective': round(avg_obj, 3)
        },
        'words': word_scores
    })

# Text Similarity
@app.route('/api/similarity/jaccard', methods=['POST'])
def jaccard_similarity_endpoint():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    similarity = intersection / union if union > 0 else 0
    
    return jsonify({
        'similarity': round(similarity, 3),
        'intersection': list(set1.intersection(set2)),
        'union_size': union
    })

@app.route('/api/similarity/cosine', methods=['POST'])
def cosine_similarity_endpoint():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    
    return jsonify({
        'similarity': round(similarity, 3),
        'vocabulary': vectorizer.get_feature_names_out().tolist()
    })

# Normalization
@app.route('/api/normalize/stem', methods=['POST'])
def stem_text():
    data = request.json
    text = data.get('text', '')
    words = text.split()
    
    porter_results = [{'word': w, 'stem': porter.stem(w)} for w in words]
    lancaster_results = [{'word': w, 'stem': lancaster.stem(w)} for w in words]
    
    return jsonify({
        'porter': porter_results,
        'lancaster': lancaster_results
    })

@app.route('/api/normalize/lemmatize', methods=['POST'])
def lemmatize_text():
    data = request.json
    text = data.get('text', '')
    
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    results = []
    for word, tag in pos_tags:
        lemma = lemmatizer.lemmatize(word.lower())
        results.append({
            'word': word,
            'pos': tag,
            'lemma': lemma
        })
    
    return jsonify({'lemmas': results})

# N-grams
@app.route('/api/ngrams/generate', methods=['POST'])
def generate_ngrams():
    data = request.json
    text = data.get('text', '')
    n = data.get('n', 2)
    
    tokens = nltk.word_tokenize(text)
    from nltk import ngrams as nltk_ngrams
    
    n_grams = list(nltk_ngrams(tokens, n))
    
    return jsonify({
        'n': n,
        'ngrams': [' '.join(gram) for gram in n_grams],
        'count': len(n_grams)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
