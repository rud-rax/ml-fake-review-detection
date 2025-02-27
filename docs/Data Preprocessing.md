To extract the specified features from the `Review` column in your dataset, we need to perform several preprocessing steps and then compute the features. Below is a Python implementation using libraries like `pandas`, `nltk`, `textstat`, and `spellchecker` for text processing and feature extraction.

### Step 1: Install Required Libraries
Make sure you have the necessary libraries installed:

```bash
pip install pandas nltk textstat pyspellchecker
```

### Step 2: Import Libraries and Load Data
```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textstat import syllable_count
from spellchecker import SpellChecker
import string

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load your dataset
df = pd.read_excel('your_file.xlsx')  # Replace with your file path
```

### Step 3: Define Functions for Feature Extraction
```python
# Initialize spell checker
spell = SpellChecker()

# Function to calculate Average Word Length (AWL)
def average_word_length(text):
    words = [word for word in word_tokenize(text) if word.isalpha()]
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)

# Function to calculate Average Sentence Length (ASL)
def average_sentence_length(text):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return 0
    return sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)

# Function to calculate Number of Words (NWO)
def number_of_words(text):
    return len([word for word in word_tokenize(text) if word.isalpha()])

# Function to calculate Number of Verbs (NVB)
def number_of_verbs(text):
    pos_tags = nltk.pos_tag(word_tokenize(text))
    return len([word for word, pos in pos_tags if pos.startswith('VB')])

# Function to calculate Number of Adjectives (NAJ)
def number_of_adjectives(text):
    pos_tags = nltk.pos_tag(word_tokenize(text))
    return len([word for word, pos in pos_tags if pos.startswith('JJ')])

# Function to calculate Number of Passive Voice (NPV)
def number_of_passive_voice(text):
    sentences = sent_tokenize(text)
    passive_count = 0
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        for i in range(len(pos_tags) - 1):
            if pos_tags[i][1] == 'VBN' and pos_tags[i + 1][1] == 'VBD':
                passive_count += 1
    return passive_count

# Function to calculate Number of Sentences (NST)
def number_of_sentences(text):
    return len(sent_tokenize(text))

# Function to calculate Content Diversity (CDV)
def content_diversity(text):
    words = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stopwords.words('english')]
    if len(words) == 0:
        return 0
    unique_words = set(words)
    return len(unique_words) / len(words)

# Function to calculate Number of Typos (NTP)
def number_of_typos(text):
    words = [word for word in word_tokenize(text) if word.isalpha()]
    misspelled = spell.unknown(words)
    return len(misspelled)

# Function to calculate Typo Ratio (TPR)
def typo_ratio(text):
    words = [word for word in word_tokenize(text) if word.isalpha()]
    if len(words) == 0:
        return 0
    misspelled = spell.unknown(words)
    return len(misspelled) / len(words)
```

### Step 4: Apply Functions to the `Review` Column
```python
# Apply the functions to the Review column
df['AWL'] = df['Review'].apply(average_word_length)
df['ASL'] = df['Review'].apply(average_sentence_length)
df['NWO'] = df['Review'].apply(number_of_words)
df['NVB'] = df['Review'].apply(number_of_verbs)
df['NAJ'] = df['Review'].apply(number_of_adjectives)
df['NPV'] = df['Review'].apply(number_of_passive_voice)
df['NST'] = df['Review'].apply(number_of_sentences)
df['CDV'] = df['Review'].apply(content_diversity)
df['NTP'] = df['Review'].apply(number_of_typos)
df['TPR'] = df['Review'].apply(typo_ratio)

# Display the DataFrame with the new features
print(df.head())
```

### Step 5: Save the Results (Optional)
If you want to save the results to a new Excel file:
```python
df.to_excel('reviews_with_features.xlsx', index=False)
```

### Explanation of Features:
1. **AWL**: Average length of words in the review.
2. **ASL**: Average number of words per sentence.
3. **NWO**: Total number of words in the review.
4. **NVB**: Number of verbs in the review.
5. **NAJ**: Number of adjectives in the review.
6. **NPV**: Number of passive voice constructions.
7. **NST**: Total number of sentences.
8. **CDV**: Ratio of unique words to total words (excluding stopwords and punctuation).
9. **NTP**: Number of misspelled words.
10. **TPR**: Ratio of misspelled words to total words.



