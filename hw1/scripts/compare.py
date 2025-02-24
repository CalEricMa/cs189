from collections import Counter

with open('spam_word_frequencies.txt', 'r') as spam_file:
    spam_words = Counter({line.split(': ')[0]: int(line.split(': ')[1]) for line in spam_file})

with open('ham_word_frequencies.txt', 'r') as ham_file:
    ham_words = Counter({line.split(': ')[0]: int(line.split(': ')[1]) for line in ham_file})

N = 50
spam_top = spam_words.most_common(N)
ham_top = ham_words.most_common(N)

unique_spam_words = [word for word, count in spam_words.items() if word not in ham_words]
unique_ham_words = [word for word, count in ham_words.items() if word not in spam_words]

print("Top words in SPAM emails:")
print(spam_top)

print("\nTop words in HAM emails:")
print(ham_top)

print("\nWords unique to SPAM:")
print(unique_spam_words[:N])

print("\nWords unique to HAM:")
print(unique_ham_words[:N])
