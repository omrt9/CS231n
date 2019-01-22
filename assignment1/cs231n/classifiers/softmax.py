import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  loss = 0.0
  for i in range(num_train):
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      
      sum_j = 0.0
      for j in range(num_classes):
          sum_j += np.exp(scores[j])
      
      loss += -correct_class_score + np.log(sum_j)
  
      for j in range(num_classes):
          dW[:, j] += (np.exp(scores[j]) * X[i])/sum_j
          if(j == y[i]):
              dW[:,y[i]] -= X[i]
    
  loss /= num_train
  dW /= num_train
  
  loss += reg * np.sum(W * W)
  dW += W*reg
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  '''num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  f = X.dot(W)
  f -= np.matrix(np.max(f, axis=1)).T
  
  temp = -f[np.arange(num_train), y]
  sum_j = np.sum(np.exp(f), axis=1)
  temp1 = np.log(sum_j)
  loss = temp + temp1
  loss /= num_train
  loss += reg * np.sum(W * W)'''

  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), y].reshape((num_train, 1))
  sum_j = np.sum(np.exp(scores), axis=1).reshape((num_train, 1))

  loss = np.sum(-1 * correct_class_scores + np.log(sum_j)) / num_train + 0.5 * reg * np.sum(W * W)

  correct_matrix = np.zeros(scores.shape)
  correct_matrix[range(num_train), y] = 1

  dW = X.T.dot(np.exp(scores) / sum_j) - X.T.dot(correct_matrix)
  dW = dW / num_train + W * reg
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

