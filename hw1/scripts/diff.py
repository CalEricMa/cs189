import pandas as pd

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
                continue
    return word_freq

ham_freq = load_word_frequencies(ham_file)
spam_freq = load_word_frequencies(spam_file)

all_words = set(ham_freq.keys()).union(set(spam_freq.keys()))

word_differences = []
for word in all_words:
    ham_count = ham_freq.get(word, 0)
    spam_count = spam_freq.get(word, 0)
    difference = abs(ham_count - spam_count)
    word_differences.append((word, ham_count, spam_count, difference))

word_differences.sort(key=lambda x: x[3], reverse=True)

df = pd.DataFrame(word_differences, columns=["Word", "Ham Frequency", "Spam Frequency", "Difference"])

print(df.head(50)) 