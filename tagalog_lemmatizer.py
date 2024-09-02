from gensim.models import Word2Vec
from collections import defaultdict

# Load model
model = Word2Vec.load("word2vec_300dim_20epochs.model")

input_word = 'ginagandahan'

# Function to load the Tagalog dictionary from a text file
def load_dictionary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words

# Function to search for subsets of a word in the dictionary
def search_subwords(input_word, dictionary):
    subwords = []
    for word in dictionary:
        it = iter(input_word)
        if all(char in it for char in word):
            subwords.append(word)
    return subwords

# Function to count the number of characters from input_word present in each word
def count_characters(word, input_word):
    return sum(1 for char in word if char in input_word)

# Function to calculate the percentage of characters from input_word present in each word
def calculate_percentage(word, input_word):
    return (count_characters(word, input_word) / len(input_word)) * 100

# Load the Tagalog dictionary text file
dictionary = load_dictionary('Filipino-wordlist.txt')

# Example usage
found_subwords = search_subwords(input_word, dictionary)

# Filter the found subwords to ensure they appear in the dictionary
filtered_word_list = [word for word in found_subwords if word in dictionary]

# Remove duplicates by converting the list to a set and back to a list
unique_filtered_word_list = list(set(filtered_word_list))

# Rank the words based on the percentage of characters present from input_word
ranked_word_list = sorted(unique_filtered_word_list, key=lambda word: calculate_percentage(word, input_word), reverse=True)

# Add percentage values to the ranked words
ranked_word_list_with_percentage = [(word, calculate_percentage(word, input_word)) for word in ranked_word_list]
print("Character Value Rank: ", ranked_word_list_with_percentage)

# Get top 100 nearest neighbors of the word
most_similar_words = model.wv.most_similar(input_word, topn=200)

# Create a dictionary of most similar words for quick lookup
similar_words_dict = {word: similarity for word, similarity in most_similar_words}
print("Similar Words: ", similar_words_dict)

# Update the ranking based on similarity
final_ranked_list = []
for word, percentage in ranked_word_list_with_percentage:
    if word in similar_words_dict:
        final_ranked_list.append((word, percentage + similar_words_dict[word] * 100))

# Sort the final ranked list based on the updated percentage
final_ranked_list = sorted(final_ranked_list, key=lambda x: x[1], reverse=True)

# Handle empty list case
if not final_ranked_list:
    final_ranked_list = [(input_word, 100.0)]

print("Lemma Basis: ", final_ranked_list)
