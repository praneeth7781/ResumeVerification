import os
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# uncomment these two if you haven't ever run these commands before, they have to be run only once
# nltk.download('punkt')
# nltk.download('stopwords')

def plot_mendenhall_curves(files_directory):
    stop_words = set(stopwords.words('english'))

    for file_name in os.listdir(files_directory):
        file_path = os.path.join(files_directory, file_name)
        with open(file_path, 'r') as file:
            content = file.read()

        # tokenization
        tokens = word_tokenize(content.lower())

        # filter out stopwords and only take alphabetical tokens
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

        # word length frequency distribution
        fd = FreqDist(len(token) for token in tokens)

        # Plot the Mendenhall's curve
        word_lengths = sorted(fd.keys())
        word_counts = [fd[word_length] for word_length in word_lengths]
        plt.plot(word_lengths, word_counts, label=file_name)

    # Set plot labels and legend
    plt.xlabel('Word Length')
    plt.ylabel('Number of Words')
    plt.legend()

    # Show the plot
    plt.show()


# Usage example
files_directory_path = '/splitup'
plot_mendenhall_curves(files_directory_path)