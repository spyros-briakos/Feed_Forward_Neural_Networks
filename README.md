# Feed_Forward_Neural_Networks
Our goal is to predict tweet comment's sentiment, from [dataset](https://drive.google.com/file/d/1dTIWNpjlrnTQBIQtaGOh0jCRYZiAQO79/view), by experimenting with two different types of vectorization (GloVe and TfIdf).

In [GloVe Model](https://github.com/spympr/Twitter_Sentiment_Classifier/blob/main/GloVe_Model.ipynb) we constructed functions which vectorize data with help of "glove.6B.50d" model (which contains 6 billion 50-dimensional pre-trained word embendding vectors), which we retrieved from [GloVe's site](https://nlp.stanford.edu/projects/glove/). Our concept was to calculate a mean vector for each tweet, in order to accelerate vectorization's procedure.Afterwards, we trained our non-deep feed forward neural network and evaluated on test data.

On the other hand, in [TfIdf Model](https://github.com/spympr/Twitter_Sentiment_Classifier/blob/main/TfIdf_Model.ipynb), we vectorized data with TdIdf-Vectorizer and we constructed a swallow and a deep feed forward neural network. After training's procedure, we displayed learning curves, classification report and ROC plot of results and finally compare our models!

In both models we managed to experiment with: 
* the number of hidden layers, and the number of their units
* the activation functions
* the loss function
* the optimizer

Note that these notebooks were implemented with ***Machine Learning Library Pytorch*** and running's procedure took place on ***Google Colab***, enhanced with ***Cuda GPU!***
