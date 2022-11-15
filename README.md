# openiit_data_analytics
# About the event:
OpenIIT is a prestigious and celebrated annual event in IIT Kharagpur and students from all disciplines and courses are open to participating in the same hence the name "open".

# The problem statement
This year's competition demanded sorting pairs of gender-biased statements from each other. The participants were provided with pairs of sentences and a label of '0' or '1' corresponding to that pair, if both the statements were gender biased or both were unbiased in that case the pair was labeled as '0' otherwise when one the sentences was biased and the other unbiased the pair was labeled '1'. The 'text_and_sorted.txt' file contained the sentences along with their indexes. The 'pairs-label-training.csv' file that I uploaded had the index of pairs of sentences and the corresponding labels.
  
# My initial approach
we first preprocessed our statements by first lemmatizing them and then removing all the standard stopwords in the English language using the nltk library.
I initially tried the problem in a way any NLP problem is treated and used embedding to embed all the words of sentences and then concatenated the embeddings of the pairs of statements and fed each matrix of embeddings to a simple deep neural network. I used google's BERT as well as TensorFlow's embedding tools. But it soon turned out that these models were not only computationally expensive but also gave only poor accuracies of around 50%.

# Realisation
I soon realized that using embeddings was not an answer as these tools were made keeping in mind to serve other purposes like completing sentences computer speech etc. as they were built to add context to statements and in our case we did not need that as we were ending up using significant computation energy trying to contextualize two concatenated statements that were independent of each other.

# My final model
So after trying out many other methodologies I decided to go with the Tf-Idf vector method. The nltk library in python provides an object named 'tfidfvectorizer' for calculating this metric and feeding the tf-IDF vector into a very simple 2-layer sequential neural network the model was able to achieve accuracy rates of 99% on the test sets. This method came with its advantages such as
  1. It did not cause excessive computation expenses like the prior models when fed into neural networks, as it did not contextualize the words in a sentence that we wanted.
  2. It worked brilliantly as I hypothesize that it took more care about the frequency of gender-biased words in a given statement and the ones with either zero or relatively many gender-sensitive words were labeled as '0' as that would happen in case of both the sentences being biased or none of them being biased and the pair of statements with a relatively intermediate number of gender-sensitive words were labeled as '1' which would have occurred in the case when one of the two sentences had to gender-sensitive words.
