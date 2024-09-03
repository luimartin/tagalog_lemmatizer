from gensim.models import Word2Vec

# Load Word2Vec model
model = Word2Vec.load("word2vec_300dim_20epochs.model")

input_word = 'pinakamakapangyarihan'

# Function to load the Tagalog dictionary from a text file
def load_dictionary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words

# Function to load morphological data
def load_morphological_data(file_path):
    morphological_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                word, root, grammatical_aspect = parts
                morphological_data[word] = root
    return morphological_data

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

# Load the morphological data
morphological_data = load_morphological_data('tagma.txt')

# Find subwords and filter them
found_subwords = search_subwords(input_word, dictionary)
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

# Define sets of prefixes, infixes, and suffixes
PREFIX_SET = [
    'nakikipag', 'pakikipag', 'pinakama', 'pagpapa', 'pinagka', 'panganga',
    'makapag', 'nakapag', 'tagapag', 'makipag', 'nakipag', 'tigapag',
    'pakiki', 'magpa', 'napaka', 'pinaka', 'ipinag', 'pagka', 
    'pinag', 'mapag', 'mapa', 'taga', 'ipag', 'tiga', 
    'pala', 'pina', 'pang', 'naka', 'nang', 'mang',
    'sing', 'ipa', 'pam', 'pan', 'pag', 'tag',
    'mai', 'mag', 'nam', 'nag', 'man', 'may',
    'ma', 'na', 'ni', 'pa', 'ka', 'um', 'in', 'i',
    'de-', 'des-', 'di-', 'ekstra-', 'elektro',
    'ikapakapagpaka', 'ikapakapagpa', 'ikapakapang', 'ikapakapag', 'ikapakapam',
    'ikapakapan', 'ikapagpaka', 'ikapakipan', 'ikapakipag', 'ikapakipam',
    'ikapakipa', 'ipakipag', 'ipagkang', 'ikapagpa', 'ikapaka', 'ikapaki',
    'ikapang', 'ipakipa', 'ikapag', 'ikapam', 'ikapan', 'ipagka', 'ipagpa',
    'ipaka', 'ipaki', 'ikapa', 'ipang', 'ikang', 'ipag', 'ikam', 'ikan', 
    'isa', 'kasing', 'kamaka', 'kanda', 'kasim', 'kasin', 'kamag',
    'kaka', 'ka', 'mangagsipagpaka', 'mangagsipag', 'mangagpaka', 'magsipagpa',
    'makapagpa', 'mangagsi', 'mangagpa', 'magsipag', 'mangagka', 'magkang',
    'magpaka', 'magpati', 'mapapag', 'mapang', 'mapasa',
    'mapapa', 'mangag', 'manga', 'magka', 'magsa',
    'mapam', 'mapan', 'maka', 'maki', 'mam',
    'nangagsipagpaka', 'nangagsipagpa', 'nagsipagpaka', 'nakapagpaka',
    'nangagsipag', 'nangagpaka', 'nangagkaka', 'nagsipagpa', 'nakapagpa',
    'nagsipag', 'nangagpa', 'nangagka', 'nangagsi',
    'napapag', 'nagpaka', 'nagpati', 'nangag', 'napasa',
    'nanga', 'nagka', 'nagpa', 'nagsa', 'nagsi', 'napag',
    'naki', 'napa', 'na', 'pagpapati', 'pagpapaka',
    'pagsasa', 'pasasa',
    'papag', 'pampa', 'panag', 'paka', 'paki',
    'pani', 'papa', 'para', 'pasa', 'pati',
]
INFIX_SET = ['um', 'in']
SUFFIX_SET = ['syon', 'dor', 'ita', 'han', 'hin', 'ing', 'ang', 'ng', 'an', 'in', 'g']

# Improved function to remove affixes from a word with hyphen handling
def remove_affixes(word):
    # Remove hyphens for easier processing
    word = word.replace('-', '')
    
    # Check if the word is in the morphological data
    if word in morphological_data:
        return morphological_data[word]
    
    original_word = word
    
    # Try to remove prefixes
    for prefix in PREFIX_SET:
        if word.startswith(prefix):
            stripped_word = word[len(prefix):]
            if stripped_word in unique_filtered_word_list:
                return stripped_word
    
    # Try to remove infixes
    for infix in INFIX_SET:
        if infix in word:
            stripped_word = word.replace(infix, '', 1)  # Remove only the first occurrence of infix
            if stripped_word in unique_filtered_word_list:
                return stripped_word
    
    # Try to remove suffixes
    for suffix in SUFFIX_SET:
        if word.endswith(suffix):
            stripped_word = word[:-len(suffix)]
            if stripped_word in unique_filtered_word_list:
                return stripped_word

    return word  # Return the word if no affixes are removed

# Remove affixes from the highest-ranked word
final_word = final_ranked_list[0][0]

# Check if the final_word is in the morphological data
if final_word in morphological_data:
    stripped_word = morphological_data[final_word]
else:
    stripped_word = remove_affixes(final_word)

# Output the final lemma
if stripped_word:
    print("Final Lemma!!: ", stripped_word)
else:
    print("Final Lemma!: ", final_word)
