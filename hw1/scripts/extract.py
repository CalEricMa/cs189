import os
import re
from collections import Counter

def extract_words_from_files(folder_path):
    """
    Extract all words from files in the specified folder and return a Counter of word frequencies.
    """
    word_counter = Counter()

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                # Read the content of the file
                content = file.read()
                # Remove special characters and split into words
                words = re.findall(r'\w+', content.lower())  # Convert to lowercase for consistent counting
                # Update the word counter
                word_counter.update(words)
    
    return word_counter

def display_top_words(word_counter, top_n=20):
    """
    Display the top N most frequent words.
    """
    print(f"Top {top_n} words in SPAM emails:")
    for word, count in word_counter.most_common(top_n):
        print(f"{word}: {count}")

def main():
    spam_folder_path = "../data/ham/"  # Replace with the path to your SPAM folder

    # Extract and count words
    word_counter = extract_words_from_files(spam_folder_path)

    # Display the most frequent words
    display_top_words(word_counter, top_n=20)

    # Save the full word frequencies to a file if needed
    with open("ham_word_frequencies.txt", "w") as output_file:
        for word, count in word_counter.most_common():
            output_file.write(f"{word}: {count}\n")

if __name__ == "__main__":
    main()
