// Example texts by task type
const examples = {
    text: {
        1: "Mark is working at Google.",
        2: "Apple Inc. was founded by Steve Jobs in California. The company is now worth over $2 trillion.",
        3: "The quick brown fox jumps over the lazy dog near the Eiffel Tower in Paris."
    },
    word: {
        1: "dog",
        2: "computer",
        3: "happiness"
    },
    twoWords: {
        1: { word1: "dog", word2: "cat" },
        2: { word1: "car", word2: "automobile" },
        3: { word1: "happy", word2: "sad" }
    },
    twoTexts: {
        1: {
            text1: "The cat sat on the mat.",
            text2: "The dog sat on the rug."
        },
        2: {
            text1: "I love programming in Python.",
            text2: "Python is my favorite programming language."
        },
        3: {
            text1: "Natural language processing is fascinating.",
            text2: "Machine learning is very interesting."
        }
    },
    ngrams: {
        1: "The quick brown fox jumps over the lazy dog.",
        2: "Natural language processing enables computers to understand human language.",
        3: "Machine learning algorithms can learn from data and make predictions."
    }
};

// Task configurations by category
const tasksByCategory = {
    basic: {
        nltk: [
            { value: 'tokenize', label: 'Tokenization', code: 'nltk_tokenize' },
            { value: 'pos_tag', label: 'POS Tagging', code: 'nltk_pos_tag' },
            { value: 'ner', label: 'Named Entity Recognition', code: 'nltk_ner' }
        ],
        spacy: [
            { value: 'tokenize', label: 'Tokenization', code: 'spacy_analyze' },
            { value: 'pos_tag', label: 'POS Tagging', code: 'spacy_analyze' },
            { value: 'ner', label: 'Named Entity Recognition', code: 'spacy_analyze' },
            { value: 'dependency', label: 'Dependency Parsing', code: 'spacy_dependency' }
        ]
    },
    semantic: {
        both: [
            { value: 'synonyms', label: 'Synonyms (WordNet)', code: 'wordnet_synonyms', inputType: 'word' },
            { value: 'hypernyms', label: 'Hypernyms (WordNet)', code: 'wordnet_hypernyms', inputType: 'word' },
            { value: 'similarity', label: 'Word Similarity', code: 'wordnet_similarity', inputType: 'twoWords' }
        ]
    },
    sentiment: {
        both: [
            { value: 'sentiment', label: 'Sentiment Analysis', code: 'sentiment' }
        ]
    },
    similarity: {
        both: [
            { value: 'jaccard', label: 'Jaccard Similarity', code: 'jaccard_similarity', inputType: 'twoTexts' },
            { value: 'cosine', label: 'Cosine Similarity', code: 'cosine_similarity', inputType: 'twoTexts' }
        ]
    },
    normalization: {
        both: [
            { value: 'stem', label: 'Stemming', code: 'stemming' },
            { value: 'lemmatize', label: 'Lemmatization', code: 'lemmatization' }
        ]
    },
    advanced: {
        both: [
            { value: 'ngrams', label: 'N-grams', code: 'ngrams', inputType: 'ngrams' }
        ]
    }
};

// Task descriptions
const taskDescriptions = {
    tokenize: 'Splits text into individual words and punctuation marks',
    pos_tag: 'Identifies the grammatical role of each word (noun, verb, etc.)',
    ner: 'Finds and classifies named entities (people, organizations, locations)',
    dependency: 'Shows grammatical relationships between words',
    synonyms: 'Finds words with similar meanings using WordNet',
    hypernyms: 'Finds broader, more general terms for a word',
    similarity: 'Computes semantic similarity between two words (0-1 scale)',
    sentiment: 'Analyzes the emotional tone of text (positive/negative/neutral)',
    jaccard: 'Measures text similarity based on shared words',
    cosine: 'Measures text similarity using vector space model',
    stem: 'Reduces words to their root form (running → run)',
    lemmatize: 'Converts words to their dictionary form',
    ngrams: 'Generates sequences of N consecutive words'
};

// DOM elements
const librarySelect = document.getElementById('library-select');
const categorySelect = document.getElementById('category-select');
const taskSelect = document.getElementById('task-select');
const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsDiv = document.getElementById('results');
const resultInfo = document.getElementById('result-info');
const loading = document.getElementById('loading');
const exampleBtns = document.querySelectorAll('.example-btn');
const codeDisplay = document.getElementById('code-display');
const codeDescription = document.getElementById('code-description');
const copyCodeBtn = document.getElementById('copy-code-btn');
const inputContainer = document.getElementById('input-container');

// Event listeners
librarySelect.addEventListener('change', updateTaskOptions);
categorySelect.addEventListener('change', updateTaskOptions);
taskSelect.addEventListener('change', () => {
    updateInputType();
    updateCodeDisplay();
    attachExampleListeners();
});
analyzeBtn.addEventListener('click', analyze);
copyCodeBtn.addEventListener('click', copyCode);

// Function to attach example button listeners
function attachExampleListeners() {
    const exampleBtns = document.querySelectorAll('.example-btn');
    exampleBtns.forEach(btn => {
        // Remove old listeners by cloning
        const newBtn = btn.cloneNode(true);
        btn.parentNode.replaceChild(newBtn, btn);

        newBtn.addEventListener('click', (e) => {
            const exampleNum = parseInt(e.target.dataset.example);
            loadExample(exampleNum);
        });
    });
}

// Load example based on current input type
function loadExample(exampleNum) {
    const selectedOption = taskSelect.options[taskSelect.selectedIndex];
    if (!selectedOption) return;

    const inputType = selectedOption.dataset.inputType || 'text';

    if (inputType === 'word') {
        const wordInput = document.getElementById('word-input');
        if (wordInput && examples.word[exampleNum]) {
            wordInput.value = examples.word[exampleNum];
        }
    } else if (inputType === 'twoWords') {
        const word1Input = document.getElementById('word1-input');
        const word2Input = document.getElementById('word2-input');
        if (word1Input && word2Input && examples.twoWords[exampleNum]) {
            word1Input.value = examples.twoWords[exampleNum].word1;
            word2Input.value = examples.twoWords[exampleNum].word2;
        }
    } else if (inputType === 'twoTexts') {
        const text1Input = document.getElementById('text1-input');
        const text2Input = document.getElementById('text2-input');
        if (text1Input && text2Input && examples.twoTexts[exampleNum]) {
            text1Input.value = examples.twoTexts[exampleNum].text1;
            text2Input.value = examples.twoTexts[exampleNum].text2;
        }
    } else if (inputType === 'ngrams') {
        const textInput = document.getElementById('text-input');
        if (textInput && examples.ngrams[exampleNum]) {
            textInput.value = examples.ngrams[exampleNum];
        }
    } else {
        // Default text input
        const textInput = document.getElementById('text-input');
        if (textInput && examples.text[exampleNum]) {
            textInput.value = examples.text[exampleNum];
        }
    }

    updateCodeDisplay();
}

// Update task options based on library and category
function updateTaskOptions() {
    const library = librarySelect.value;
    const category = categorySelect.value;

    taskSelect.innerHTML = '';

    let tasks = [];
    if (tasksByCategory[category]) {
        if (tasksByCategory[category][library]) {
            tasks = tasksByCategory[category][library];
        } else if (tasksByCategory[category].both) {
            tasks = tasksByCategory[category].both;
        }
    }

    tasks.forEach(task => {
        const option = document.createElement('option');
        option.value = task.value;
        option.textContent = task.label;
        option.dataset.code = task.code;
        option.dataset.inputType = task.inputType || 'text';
        taskSelect.appendChild(option);
    });

    updateInputType();
    updateCodeDisplay();
}

// Update input fields based on task type
function updateInputType() {
    const selectedOption = taskSelect.options[taskSelect.selectedIndex];
    if (!selectedOption) return;

    const inputType = selectedOption.dataset.inputType || 'text';

    inputContainer.innerHTML = '';

    if (inputType === 'word') {
        inputContainer.innerHTML = `
            <div class="input-group">
                <label>Word</label>
                <input type="text" id="word-input" placeholder="Enter a word..." value="dog">
            </div>
        `;
    } else if (inputType === 'twoWords') {
        inputContainer.innerHTML = `
            <div class="input-row">
                <div class="input-group">
                    <label>Word 1</label>
                    <input type="text" id="word1-input" placeholder="First word..." value="dog">
                </div>
                <div class="input-group">
                    <label>Word 2</label>
                    <input type="text" id="word2-input" placeholder="Second word..." value="cat">
                </div>
            </div>
        `;
    } else if (inputType === 'twoTexts') {
        inputContainer.innerHTML = `
            <div class="input-group">
                <label>Text 1</label>
                <textarea id="text1-input" placeholder="First text...">The cat sat on the mat.</textarea>
            </div>
            <div class="input-group">
                <label>Text 2</label>
                <textarea id="text2-input" placeholder="Second text...">The dog sat on the rug.</textarea>
            </div>
        `;
    } else if (inputType === 'ngrams') {
        inputContainer.innerHTML = `
            <div class="input-group">
                <label>Text</label>
                <textarea id="text-input" placeholder="Enter your text...">The quick brown fox jumps over the lazy dog.</textarea>
            </div>
            <div class="input-group">
                <label>N (number of words per gram)</label>
                <input type="number" id="n-input" min="1" max="5" value="2">
            </div>
        `;
    } else {
        inputContainer.innerHTML = `
            <textarea id="text-input" placeholder="Enter your text...">Mark is working at Google.</textarea>
        `;
    }
}

// Update code display
async function updateCodeDisplay() {
    const selectedOption = taskSelect.options[taskSelect.selectedIndex];
    if (!selectedOption) return;

    const codeTemplate = selectedOption.dataset.code;
    const task = selectedOption.value;

    // Update description
    codeDescription.textContent = taskDescriptions[task] || 'Python code for this task';

    // Get input data
    const inputData = getInputData();

    try {
        const response = await fetch(`/api/code/${codeTemplate}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });

        const data = await response.json();
        codeDisplay.textContent = data.code || '# Code not available';
    } catch (error) {
        console.error('Error fetching code:', error);
        codeDisplay.textContent = '# Error loading code';
    }
}

// Get input data based on current input type
function getInputData() {
    const selectedOption = taskSelect.options[taskSelect.selectedIndex];
    const inputType = selectedOption ? selectedOption.dataset.inputType : 'text';

    const data = {};

    if (inputType === 'word') {
        const wordInput = document.getElementById('word-input');
        data.word = wordInput ? wordInput.value : 'dog';
    } else if (inputType === 'twoWords') {
        const word1Input = document.getElementById('word1-input');
        const word2Input = document.getElementById('word2-input');
        data.word1 = word1Input ? word1Input.value : 'dog';
        data.word2 = word2Input ? word2Input.value : 'cat';
    } else if (inputType === 'twoTexts') {
        const text1Input = document.getElementById('text1-input');
        const text2Input = document.getElementById('text2-input');
        data.text1 = text1Input ? text1Input.value : 'The cat sat on the mat.';
        data.text2 = text2Input ? text2Input.value : 'The dog sat on the rug.';
    } else if (inputType === 'ngrams') {
        const textInput = document.getElementById('text-input');
        const nInput = document.getElementById('n-input');
        data.text = textInput ? textInput.value : 'The quick brown fox';
        data.n = nInput ? parseInt(nInput.value) : 2;
    } else {
        const textInput = document.getElementById('text-input');
        data.text = textInput ? textInput.value : 'Mark is working at Google.';
    }

    return data;
}

// Copy code to clipboard
function copyCode() {
    const code = codeDisplay.textContent;
    navigator.clipboard.writeText(code).then(() => {
        copyCodeBtn.textContent = '✓ Copied!';
        setTimeout(() => {
            copyCodeBtn.textContent = 'Copy';
        }, 2000);
    });
}

// Main analyze function
async function analyze() {
    const library = librarySelect.value;
    const category = categorySelect.value;
    const task = taskSelect.value;
    const inputData = getInputData();

    showLoading(true);

    try {
        let endpoint = '';
        let requestData = inputData;

        // Determine endpoint
        if (category === 'basic') {
            if (library === 'nltk') {
                endpoint = `/api/nltk/${task}`;
            } else if (library === 'spacy') {
                endpoint = task === 'dependency' ? '/api/spacy/dependency_tree' : '/api/spacy/analyze';
            }
        } else if (category === 'semantic') {
            endpoint = `/api/wordnet/${task}`;
        } else if (category === 'sentiment') {
            endpoint = '/api/sentiment/analyze';
        } else if (category === 'similarity') {
            endpoint = `/api/similarity/${task}`;
        } else if (category === 'normalization') {
            endpoint = `/api/normalize/${task}`;
        } else if (category === 'advanced') {
            endpoint = `/api/ngrams/generate`;
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        displayResults(data, library, task, category);

    } catch (error) {
        console.error('Error:', error);
        resultsDiv.innerHTML = `
            <div class="empty-state">
                <p style="color: #dc3545;">❌ Error: ${error.message}</p>
                <p>Make sure the server is running and dependencies are installed.</p>
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

// Display results based on task type
function displayResults(data, library, task, category) {
    resultsDiv.innerHTML = '';
    resultInfo.textContent = '';

    if (category === 'basic') {
        if (task === 'tokenize') {
            displayTokens(data, library);
        } else if (task === 'pos_tag') {
            displayPOSTags(data, library);
        } else if (task === 'ner') {
            displayEntities(data, library);
        } else if (task === 'dependency') {
            displayDependencyTree(data);
        }
    } else if (category === 'semantic') {
        if (task === 'synonyms') {
            displaySynonyms(data);
        } else if (task === 'hypernyms') {
            displayHypernyms(data);
        } else if (task === 'similarity') {
            displayWordSimilarity(data);
        }
    } else if (category === 'sentiment') {
        displaySentiment(data);
    } else if (category === 'similarity') {
        displayTextSimilarity(data, task);
    } else if (category === 'normalization') {
        if (task === 'stem') {
            displayStemming(data);
        } else if (task === 'lemmatize') {
            displayLemmatization(data);
        }
    } else if (category === 'advanced') {
        displayNgrams(data);
    }
}

// Display functions (keeping existing ones and adding new ones)
function displayTokens(data, library) {
    const tokens = library === 'nltk' ? data.tokens : data.tokens;
    const count = library === 'nltk' ? data.count : tokens.length;

    resultInfo.textContent = `${count} tokens found`;

    const container = document.createElement('div');
    container.className = 'tokens-container';

    tokens.forEach(token => {
        const tokenEl = document.createElement('div');
        tokenEl.className = 'token';
        tokenEl.textContent = token;
        container.appendChild(tokenEl);
    });

    resultsDiv.appendChild(container);
}

function displayPOSTags(data, library) {
    const posTags = data.pos_tags;
    resultInfo.textContent = `${posTags.length} tokens tagged`;

    const table = document.createElement('table');
    table.className = 'pos-table';

    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>#</th>
            <th>Word</th>
            <th>POS Tag</th>
            ${library === 'spacy' ? '<th>POS Type</th>' : ''}
        </tr>
    `;
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    posTags.forEach((item, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td>
            <td style="font-family: Consolas, monospace; font-weight: 600;">${item.word}</td>
            <td><span class="pos-tag">${item.tag}</span></td>
            ${library === 'spacy' ? `<td>${item.pos}</td>` : ''}
        `;
        tbody.appendChild(row);
    });
    table.appendChild(tbody);

    resultsDiv.appendChild(table);
}

function displayEntities(data, library) {
    const entities = data.entities;

    if (entities.length === 0) {
        resultsDiv.innerHTML = '<div class="empty-state"><p>No named entities found</p></div>';
        return;
    }

    resultInfo.textContent = `${entities.length} entities found`;

    const container = document.createElement('div');
    container.className = 'entities-container';

    entities.forEach(entity => {
        const entityEl = document.createElement('div');
        entityEl.className = 'entity';
        entityEl.innerHTML = `
            <div class="entity-text">${entity.text}</div>
            <div class="entity-label">${entity.label}</div>
        `;
        container.appendChild(entityEl);
    });

    resultsDiv.appendChild(container);
}

function displayDependencyTree(data) {
    const { nodes, edges } = data;
    resultInfo.textContent = `${nodes.length} nodes, ${edges.length} dependencies`;

    const container = document.createElement('div');
    container.className = 'tree-container';

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'tree-svg');

    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
        <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#999" />
        </marker>
    `;
    svg.appendChild(defs);

    const nodeWidth = 120;
    const horizontalSpacing = 20;
    const totalWidth = nodes.length * (nodeWidth + horizontalSpacing);
    const totalHeight = 400;

    svg.setAttribute('width', totalWidth);
    svg.setAttribute('height', totalHeight);
    svg.setAttribute('viewBox', `0 0 ${totalWidth} ${totalHeight}`);

    const nodePositions = {};
    nodes.forEach((node, index) => {
        const x = index * (nodeWidth + horizontalSpacing) + nodeWidth / 2;
        const y = totalHeight - 100;
        nodePositions[node.id] = { x, y };
    });

    edges.forEach(edge => {
        const source = nodePositions[edge.source];
        const target = nodePositions[edge.target];

        if (source && target) {
            const midY = (source.y + target.y) / 2 - 50;
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            const d = `M ${source.x} ${source.y - 30} Q ${(source.x + target.x) / 2} ${midY} ${target.x} ${target.y - 30}`;
            path.setAttribute('d', d);
            path.setAttribute('class', 'tree-edge');
            svg.appendChild(path);

            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', (source.x + target.x) / 2);
            label.setAttribute('y', midY - 5);
            label.setAttribute('class', 'edge-label');
            label.setAttribute('text-anchor', 'middle');
            label.textContent = edge.label;
            svg.appendChild(label);
        }
    });

    nodes.forEach(node => {
        const pos = nodePositions[node.id];
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pos.x);
        circle.setAttribute('cy', pos.y);
        circle.setAttribute('r', 25);
        circle.setAttribute('class', 'node-circle');
        g.appendChild(circle);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', pos.x);
        text.setAttribute('y', pos.y + 5);
        text.setAttribute('class', 'node-text');
        text.setAttribute('text-anchor', 'middle');
        text.textContent = node.text;
        g.appendChild(text);

        const tag = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        tag.setAttribute('x', pos.x);
        tag.setAttribute('y', pos.y + 45);
        tag.setAttribute('class', 'node-tag');
        tag.setAttribute('text-anchor', 'middle');
        tag.textContent = node.tag;
        g.appendChild(tag);

        svg.appendChild(g);
    });

    container.appendChild(svg);
    resultsDiv.appendChild(container);
}

function displaySynonyms(data) {
    const synsets = data.synsets;

    if (synsets.length === 0) {
        resultsDiv.innerHTML = '<div class="empty-state"><p>No synsets found for this word</p></div>';
        return;
    }

    resultInfo.textContent = `${synsets.length} synsets found for "${data.word}"`;

    const container = document.createElement('div');
    container.className = 'synset-container';

    synsets.forEach(syn => {
        const synEl = document.createElement('div');
        synEl.className = 'synset';

        const lemmasHTML = syn.lemmas.map(l => `<span class="lemma">${l}</span>`).join('');

        synEl.innerHTML = `
            <div class="synset-name">${syn.synset}</div>
            <div class="synset-definition">${syn.definition}</div>
            <div class="synset-lemmas">${lemmasHTML}</div>
        `;

        container.appendChild(synEl);
    });

    resultsDiv.appendChild(container);
}

function displayHypernyms(data) {
    const hypernyms = data.hypernyms;

    if (hypernyms.length === 0) {
        resultsDiv.innerHTML = '<div class="empty-state"><p>No hypernyms found</p></div>';
        return;
    }

    resultInfo.textContent = `${hypernyms.length} hypernyms found for "${data.word}"`;

    const container = document.createElement('div');
    container.className = 'synset-container';

    hypernyms.forEach(hyp => {
        const hypEl = document.createElement('div');
        hypEl.className = 'synset';
        hypEl.innerHTML = `
            <div class="synset-name">${hyp.name}</div>
            <div class="synset-definition">${hyp.definition}</div>
        `;
        container.appendChild(hypEl);
    });

    resultsDiv.appendChild(container);
}

function displayWordSimilarity(data) {
    const similarity = data.similarity;

    const container = document.createElement('div');
    container.className = 'similarity-result';
    container.innerHTML = `
        <div class="similarity-value">${similarity !== null ? similarity.toFixed(3) : 'N/A'}</div>
        <div class="similarity-label">Similarity between "${data.word1}" and "${data.word2}"</div>
    `;

    resultsDiv.appendChild(container);
}

function displaySentiment(data) {
    const overall = data.overall;

    const container = document.createElement('div');

    const overallDiv = document.createElement('div');
    overallDiv.className = 'sentiment-overall';
    overallDiv.innerHTML = `
        <div class="sentiment-score positive">
            <div class="sentiment-label">Positive</div>
            <div class="sentiment-value">${overall.positive.toFixed(3)}</div>
        </div>
        <div class="sentiment-score negative">
            <div class="sentiment-label">Negative</div>
            <div class="sentiment-value">${overall.negative.toFixed(3)}</div>
        </div>
        <div class="sentiment-score neutral">
            <div class="sentiment-label">Objective</div>
            <div class="sentiment-value">${overall.objective.toFixed(3)}</div>
        </div>
    `;

    container.appendChild(overallDiv);

    if (data.words && data.words.length > 0) {
        const table = document.createElement('table');
        table.className = 'pos-table';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Word</th>
                    <th>Positive</th>
                    <th>Negative</th>
                    <th>Objective</th>
                </tr>
            </thead>
            <tbody>
                ${data.words.map(w => `
                    <tr>
                        <td style="font-family: Consolas, monospace;">${w.word}</td>
                        <td>${w.pos}</td>
                        <td>${w.neg}</td>
                        <td>${w.obj}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;
        container.appendChild(table);
    }

    resultsDiv.appendChild(container);
}

function displayTextSimilarity(data, task) {
    const container = document.createElement('div');
    container.className = 'similarity-result';
    container.innerHTML = `
        <div class="similarity-value">${data.similarity.toFixed(3)}</div>
        <div class="similarity-label">${task === 'jaccard' ? 'Jaccard' : 'Cosine'} Similarity</div>
    `;

    resultsDiv.appendChild(container);
}

function displayStemming(data) {
    const container = document.createElement('div');

    const porterTable = createStemmingTable('Porter Stemmer', data.porter);
    const lancasterTable = createStemmingTable('Lancaster Stemmer', data.lancaster);

    container.appendChild(porterTable);
    container.appendChild(document.createElement('br'));
    container.appendChild(lancasterTable);

    resultsDiv.appendChild(container);
}

function createStemmingTable(title, results) {
    const div = document.createElement('div');
    div.innerHTML = `<h3 style="margin-bottom: 10px;">${title}</h3>`;

    const table = document.createElement('table');
    table.className = 'pos-table';
    table.innerHTML = `
        <thead>
            <tr>
                <th>Original</th>
                <th>Stem</th>
            </tr>
        </thead>
        <tbody>
            ${results.map(r => `
                <tr>
                    <td style="font-family: Consolas, monospace;">${r.word}</td>
                    <td style="font-family: Consolas, monospace; font-weight: bold;">${r.stem}</td>
                </tr>
            `).join('')}
        </tbody>
    `;

    div.appendChild(table);
    return div;
}

function displayLemmatization(data) {
    const table = document.createElement('table');
    table.className = 'pos-table';
    table.innerHTML = `
        <thead>
            <tr>
                <th>Word</th>
                <th>POS</th>
                <th>Lemma</th>
            </tr>
        </thead>
        <tbody>
            ${data.lemmas.map(l => `
                <tr>
                    <td style="font-family: Consolas, monospace;">${l.word}</td>
                    <td><span class="pos-tag">${l.pos}</span></td>
                    <td style="font-family: Consolas, monospace; font-weight: bold;">${l.lemma}</td>
                </tr>
            `).join('')}
        </tbody>
    `;

    resultsDiv.appendChild(table);
}

function displayNgrams(data) {
    resultInfo.textContent = `${data.count} ${data.n}-grams generated`;

    const container = document.createElement('div');
    container.className = 'ngrams-container';

    data.ngrams.forEach(ngram => {
        const ngramEl = document.createElement('div');
        ngramEl.className = 'ngram';
        ngramEl.textContent = ngram;
        container.appendChild(ngramEl);
    });

    resultsDiv.appendChild(container);
}

function showLoading(show) {
    if (show) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}

// Initialize
updateTaskOptions();
updateCodeDisplay();
attachExampleListeners();
