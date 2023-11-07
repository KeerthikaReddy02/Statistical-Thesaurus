# Statistical-Thesaurus
Statistical Thesaurus is an IER Model for query expansion

# Algorithm
1)	Import all of the necessary libraries.
2)	Tokenize the documents and the query.
3)	Perform pre-processing by doing lemmatization, stemming and converting into lowercase for the tokens in the documents and the query.
4)	Make a set of all the words in the documents.
5)	Find the frequency of each word in each document.
6)	Find the TF for each word in all documents.
7)	Find the IDF for all of the words.
8)	Find the weights for each word in all the documents.
9)	Make a term-doc matrix that stores the weights of all the words in all the documents.
10)	 Find the similarity between each pair of documents to get a similarity 2D matrix of number of documents * number of documents.
11)	 Find the pair of documents that are the most similar to each other.
12)	 Merge the 2 most similar documents together using complete link algorithm.
13)	 Remove the weights of the 2 similar documents from the term-doc matrix and add the new weights vector to the term-doc matrix.
14)	 Go back to step 10 until the stopCriterion is met, in this code assumed as 3.
15)	 Take the final clusters and find the similarity between the query and each of the clusters.
16)	 Choose the cluster which has maximum similarity.
17)	 Make a dictionary of the word:weight for each word in the most similar cluster.
18)	 Sort the words in the decreasing order of the weights.
19)	 Choose the top ‘n’ words here taken as 5 which are not already in the query.
20)	 Append these top words to the original query and print it.

# Output screenshot
<img width="570" alt="image" src="https://github.com/KeerthikaReddy02/Statistical-Thesaurus/assets/78225681/8d1daade-565c-4032-ae88-c746bdc20857">


# Instructions to execute the program
1) Download the StatThesaurus.py file
2) pip install nltk
3) Change the documents if required
4) Run the program
