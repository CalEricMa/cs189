import os
import re
from collections import Counter

# Define paths to SPAM and HAM email directories
SPAM_DIR = "../data/spam/"
HAM_DIR = "../data/ham/"

def preprocess_text(text):
    """ Tokenize and clean text: remove non-alphabetic characters & lowercase """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words

def get_word_frequencies(directory):
    """ Read all text files in a directory and count word frequencies """
    word_counter = Counter()
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                words = preprocess_text(file.read())
                word_counter.update(words)
    
    return word_counter

# Get word frequencies for SPAM and HAM emails
spam_word_counts = get_word_frequencies(SPAM_DIR)
ham_word_counts = get_word_frequencies(HAM_DIR)

# Compute absolute frequency difference
word_diff = Counter()
all_words = set(spam_word_counts.keys()).union(set(ham_word_counts.keys()))

for word in all_words:
    diff = abs(spam_word_counts[word] - ham_word_counts[word])
    word_diff[word] = diff

# Get the top 50 words with the highest absolute difference
top_diff_words = word_diff.most_common(50)

# Print results
print("\nTop 50 Most Differentiating Words Between SPAM and HAM:")
for word, diff in top_diff_words:
    spam_count = spam_word_counts[word]
    ham_count = ham_word_counts[word]
    print(f"{word}: SPAM={spam_count}, HAM={ham_count}, Difference={diff}")
