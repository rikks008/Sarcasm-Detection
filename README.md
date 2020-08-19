Sarcasm Detection Using Different Embeddings

OVERVIEW OF DATASET

Past studies in Sarcasm Detection mostly make use of Twitter datasets collected using hashtag based supervision but such datasets are noisy in terms of labels and language. Furthermore, many tweets are replies to other tweets and detecting sarcasm in these requires the availability of contextual tweets.
To overcome the limitations related to noise in Twitter datasets, this News Headlines dataset for Sarcasm Detection is collected from two news website. TheOnion aims at producing sarcastic versions of current events and we collected all the headlines from News in Brief and News in Photos categories (which are sarcastic). We collect real (and non-sarcastic) news headlines from HuffPost.
The dataset consists about 28000 text data points where each data category belongs to 2 category - Sarcastic or Not Sarcastic
I had used two models for making predictions - Word2Vec and GloVe Embeddings. We will then compare their results and see which performs better

1.DATA VISUALIZATION 
  * Word Clouds
  * Plots 

2.DATA PREPROCESSING 
  * Removing Stopwords
  * Removing HTML tags
  * Removing any special characters
 
3. INTRODUCTION TO WORD EMBEDDINGS 
   
   WHAT IS WORD EMBEDDINGS 
   
   Word embeddings is a class of techniques where individual words are represented as real-valued vectors in a predefined vector space. Each word is mapped to one                  vector and the vector values are learned in a way that resembles a neural network, and hence the technique is often lumped into the field of deep learning.
   
   Key to the approach is the idea of using a dense distributed representation for each word.
   Each word is represented by a real-valued vector, often tens or hundreds of dimensions. This is contrasted to the thousands or millions of dimensions required for sparse word    representations, such as a one-hot encoding.
   
   WHY DO WE NEED WORD EMBEDDING
   
   The way which words are represented to the computer is in the form of word vectors. One of the simplest forms of word vectors is one-hot encoded vectors. 
   The vector consists of 0s in all cells with the exception of a single 1 in a cell used uniquely to identify the word.
   Using encodings like this do not capture anything apart from the presence and absence of words in a sentence. Bag of Words is one such method.
   It is a popular technique for feature extraction from text. Bag of word model processes the text to find how many times each word appeared in the sentence. This is also          called as vectorization.
   
   Problem with Bag of Words
   
    1.In the bag of words model, each document is represented as a word-count vector. These counts can be binary counts, a word may occur in the text or not or will have               absolute counts. The size of the vector is equal to the number of elements in the vocabulary. If most of the elements are zero then the bag of words will be a sparse           matrix.In deep learning, we would have sparse matrix as we will be working with huge amount of training data. Sparse representations are harder to model both for                 computational reasons as well as for informational reasons.
    
    2.Huge amount of weights: Huge input vectors means a huge number of weights for a neural network.
    
    3.Computationally intensive: More weights means more computation required to train and predict.
    
    4.Lack of meaningful relations and no consideration for order of words: BOW is a collection of words that appear in the text or sentences with the word counts. Bag of words       does not take into consideration the order in which they appear.
    
 WORD EMBEDDINGS IS THE SOLUTION

  Embeddings translate large sparse vectors into a lower-dimensional space that preserves semantic relationships.
  Word embeddings is a technique where individual words of a domain or language are represented as real-valued vectors in a lower dimensional space.
  Sparse Matrix problem with BOW is solved by mapping high-dimensional data into a lower-dimensional space.
  Lack of meaningful relationship issue of BOW is solved by placing vectors of semantically similar items close to each other. This way words that have similar meaning have       similar distances in the vector space.

   

4.USING WORD2VEC
  Word2Vec is a statistical method for efficiently learning a standalone word embedding from a text corpus.

  It was developed by Tomas Mikolov, et al. at Google in 2013 as a response to make the neural-network-based training of the embedding more efficient and since then has become     the de facto standard for developing pre-trained word embedding.
  Word2vec models are shallow neural network with an input layer, a projection layer and an output layer. It is trained to reconstruct linguistic contexts of words. Input layer   for Word2vec neural network takes a larger corpus of text to produce a vector space, typically of several hundred dimensions. Every unique word in the text corpus is assigned   a corresponding vector in the space.
  
  Two different learning models were introduced that can be used as part of the word2vec approach to learn the word embedding; they are:
  
  1.Continuous Bag-of-Words, or CBOW model.
  
  2.Continuous Skip-Gram Model.
  
  The CBOW model learns the embedding by predicting the current word based on its context.
  The continuous skip-gram model learns by predicting the surrounding words given a current word.

 
5.USING GLOVE
  
  GloVe was developed by Pennington, et al. at Stanford. It is called Global Vectors as the global corpus statistics are captured directly by the model.
  It leverages both
  
  * Global matrix factorization methods like latent semantic analysis (LSA) for generating low-dimensional word representations
  * Local context window methods such as the skip-gram model of Mikolov et al  

6.COMPARISION OF BOTH THE MODELS
  * Accuracy and Loss PLots

7.RESULT ANALYSIS
  * Confusion Matrix 

