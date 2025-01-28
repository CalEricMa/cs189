import pandas as pd

# Load the data from files
ham_file = "ham_word_frequencies.txt"
spam_file = "spam_word_frequencies.txt"

def load_word_frequencies(file_path):
    word_freq = {}
    with open(file_path, "r") as file:
        for line in file:
            try:
                word, freq = line.strip().split(": ")
                word_freq[word] = int(freq)
            except ValueError:
                continue  # Skip malformed lines
    return word_freq

ham_freq = load_word_frequencies(ham_file)
spam_freq = load_word_frequencies(spam_file)

# Combine keys from both files
all_words = set(ham_freq.keys()).union(set(spam_freq.keys()))

# Calculate differences
word_differences = []
for word in all_words:
    ham_count = ham_freq.get(word, 0)
    spam_count = spam_freq.get(word, 0)
    difference = abs(ham_count - spam_count)
    word_differences.append((word, ham_count, spam_count, difference))

# Sort by the largest difference
word_differences.sort(key=lambda x: x[3], reverse=True)

# Convert to a DataFrame for better display
df = pd.DataFrame(word_differences, columns=["Word", "Ham Frequency", "Spam Frequency", "Difference"])

# Display the top results
print(df.head(50))  # Top 20 words with the biggest differences
