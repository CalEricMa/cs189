import os
import re
from collections import Counter

# Define paths to SPAM and HAM email directories
SPAM_DIR = "../data/spam/"
HAM_DIR = "../data/ham/"
OUTPUT_FILE = "spam_ham_word_analysis.txt"

def get_spam_dominant_words(spam_counts, ham_counts, top_n=20):
    """ Get the top N words where SPAM count is greater than HAM count, sorted by difference """
    spam_dominant = {
        word: (spam_counts[word], ham_counts[word], spam_counts[word] - ham_counts[word])
        for word in spam_counts if spam_counts[word] > ham_counts[word]
    }
    # Sort by difference in descending order
    sorted_spam_dominant = sorted(spam_dominant.items(), key=lambda x: x[1][2], reverse=True)[:top_n]
    return sorted_spam_dominant


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

# Get the top 50 words in SPAM
top_spam_words = spam_word_counts.most_common(50)

# Get the top 50 words in HAM
top_ham_words = ham_word_counts.most_common(50)

# Get the top 20 words where SPAM outnumbers HAM
top_spam_dominant_words = get_spam_dominant_words(spam_word_counts, ham_word_counts, top_n=20)

# Write results to a file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\nTop 50 Most Differentiating Words Between SPAM and HAM:\n")
    for word, diff in top_diff_words:
        spam_count = spam_word_counts[word]
        ham_count = ham_word_counts[word]
        f.write(f"{word}: SPAM={spam_count}, HAM={ham_count}, Difference={diff}\n")

    f.write("\nTop 50 Words in SPAM Emails:\n")
    for word, count in top_spam_words:
        f.write(f"{word}: {count}\n")

    f.write("\nTop 50 Words in HAM Emails:\n")
    for word, count in top_ham_words:
        f.write(f"{word}: {count}\n")

    f.write("\nTop 20 Words Where SPAM Outnumbers HAM:\n")
    for word, (spam_count, ham_count, diff) in top_spam_dominant_words:
        f.write(f"{word}: SPAM={spam_count}, HAM={ham_count}, Difference={diff}\n")

print(f"Analysis complete! Results saved to {OUTPUT_FILE}")
