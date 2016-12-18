import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
# from collections import Counter
import pandas as pd

# Global variables to tweak various settings
TRAINING_DATA_FILENAME = "TrainDataset.tsv"  # Name of the TSV file from which to read labelled training data
TEST_DATA_FILENAME = "TestDataset.tsv"  # Name of the TSV file from which to read unlabelled test data
OUTPUT_FILENAME = "LabelledOutput.csv"  # Name of the CSV output file to write
ID_HEADER = "document_id"  # Name of the column header that identifies documents for the output file
ESTIMATORS = 500  # How many estimator trees to use for the Random Forest model
# 100 estimators is a good base, 250 sees about 0.5% improvement, 500 gains another about 0.2%


# Spell checking related code from http://norvig.com/spell-correct.html
# Omitted all code here for now as it is crazy slow with the huge datasets
'''
def words(text): return re.findall(r'\w+', text.lower())

print "Reading big text file for spell checking purposes"
WORDS = Counter(words(open('big.txt').read()))


def P(word, N=sum(WORDS.values())):
    """Probability of `word`."""
    return WORDS[word] / N


def correction(word):
    """Most probable spelling correction for word."""
    return max(candidates(word), key=P)


def candidates(word):
    """Generate possible spelling corrections for word."""
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]


def known(words_passed):
    """The subset of `words` that appear in the dictionary of WORDS."""
    return set(w for w in words_passed if w in WORDS)


def edits1(word):
    """All edits that are one edit away from `word`."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# End of spell checking related code library


# My function to return the suggested word correction or original if same
def spelling_function(word):
    corrected = correction(word)
    if corrected == word:
        return word
    else:
        return corrected
'''


# Converts a review to a string of words after cleaning
def review_to_words(raw_review):
    # Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # Convert to lower case and split into words
    lower_words = letters_only.lower().split()

    # The spell checking may improve the result but is way too ludicrously slow to use with this data
    '''
    corrected_words = []
    for w in words:
        corrected_words.append(spelling_function(w))
    '''

    # Convert stop word list to set for speed
    stops = set(stopwords.words("english"))
    # Remove stop words
    meaningful_words = [w for w in lower_words if w not in stops]

    # Stemming actually seemed to make the program perform slightly worse, so it has been omitted
    '''
    ps = PorterStemmer()
    stemmed_words = []
    for w in meaningful_words:
        stemmed_words.append(ps.stem(w))
    '''

    # Join the words back into a string and return it
    return " ".join(meaningful_words)


# Trains the Random Forest model with a Bag of Words from the given labelled training data
def train_function():
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size

    print "Cleaning and parsing the training set movie reviews...\n"
    clean_train_reviews = []
    for i in xrange(0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        if (i + 1) % 1000 == 0:
            print "Training data review %d of %d\n" % (i + 1, num_reviews)
        clean_train_reviews.append(review_to_words(train["review"][i]))

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    # Initialize a Random Forest classifier with 100 trees
    random_forest = RandomForestClassifier(n_estimators=ESTIMATORS)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    print "Fitting the random forest model to the bag of words, this may take a while..."
    random_forest = random_forest.fit(train_data_features, train["sentiment"])

    return random_forest


# Labels the given test data using the trained Random Forest model
def test_function(passed_forest):

    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = []

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0, num_reviews):
        if (i + 1) % 1000 == 0:
            print "Test data review %d of %d\n" % (i + 1, num_reviews)
        clean_review = review_to_words(test["review"][i])
        clean_test_reviews.append(clean_review)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print "Predicting the sentiments using the Random Forest model, this shouldn't take too long..."
    test_result = passed_forest.predict(test_data_features)

    # Return the labelled sentiments
    return test_result


# Prepares and writes a comma-separated output file with the given test results
def output_function(result_passed):
    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    print "Preparing the output file"
    output = pd.DataFrame(data={ID_HEADER: test[ID_HEADER], "sentiment": result_passed})

    # Use pandas to write the comma-separated output file
    print "Writing the output file"
    output.to_csv(OUTPUT_FILENAME, index=False, quoting=3)


if __name__ == "__main__":
    print "Creating the Bag of Words vectorizer"
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

    print "Reading Training Data"
    train = pd.read_csv(TRAINING_DATA_FILENAME, header=0, delimiter="\t", quoting=3)
    forest = train_function()

    print "Reading Test Data"
    test = pd.read_csv(TEST_DATA_FILENAME, header=0, delimiter="\t", quoting=3)
    result = test_function(forest)

    output_function(result)
    print "Program finished successfully"
