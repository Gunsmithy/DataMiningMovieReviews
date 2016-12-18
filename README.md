# DataMiningMovieReviews
Sentiment Analysis of Movie Reviews using Random Forest on Bag of Words

Submitted in partial fulfillment for the SOFE 4870U - Data Mining, Kaggle-based assignments at UOIT.

My code takes the labelled training data of movie reviews and attempts to first clean the data before creating a Bag of Words to train a Random Forest model.
The cleaning removes any HTML, non-letters, stop words, and sets it all to lowercase.
Stemming was explored but turned out to have a negligible or even negative impact on the score.
Spell checking was also implemented to possibly increase score but the execution time on such a large dataset was not feasible for submission.
The file used for the spell-checking 'big.txt' was omitted from this for size but can be found along with the spellchecker [on the creator's site.](http://norvig.com/spell-correct.html)
PyEnchant was not able to be used for spell-checking due to the memory limitations of the 32-bit Python interpreter it requires.

This code was run using Python 2.7.12 64-bit on Windows 10.
To use the program, simply install the requirements using 'pip install -r requirements.txt' and run 'python MovieReviews.py'
If also using Windows, Wheels can be found for scipy and numpy [on this site:](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

Global variables at the top of the script can be changed to set the input and output file names and format.
The number of estimators used by the Random Tree model can also be increased for slightly improved accuracy at the cost of execution time.
