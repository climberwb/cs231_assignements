import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  

  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        temp_scores = scores
        # temp_scores[j] = 0
        margin = temp_scores - correct_class_score + 1
        margin[j] = 0
        
        m_total = sum([1 for m in margin if m>0])
        # print margin
        # if margin >0:
        dW[:,j] -= m_total*X[i,:].T
        # continue
      else:
        margin = scores[j] - correct_class_score + 1 # note delta = 
        
        # from IPython.core.debugger import Tracer
        # Tracer()() #this one triggers the debugger
      
        if margin > 0:
          loss += margin
          dW[:,j] += X[i,:].T
      
    
        
        
       
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # print (reg*np.sum(W,axis=0 )).reshape(1,10).shape
  # print dW.shape
  dW = (dW/num_train)+reg*W
   ##### added gradient code below
  # http://cs231n.github.io/optimization-1/#vis
  # h=0.001
  # dW[i] = (margin+1 - fx) / h
   ###########
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  
  scores = X.dot(W)
  
    
  yi = np.choose(y, scores.T)
  difference_of_scores_plus_one = (scores-yi[:,None]+1)
  difference_of_scores_plus_one[np.arange(X.shape[0]), y] = -np.inf
  # margins_plus_1 = np.sum(difference_of_scores_plus_one, axis=1)
  # margins = margins_plus_1 -1
  # print margins[margins>0]
  margins_greater_than_zero = difference_of_scores_plus_one[difference_of_scores_plus_one>0]
  loss = np.sum(margins_greater_than_zero)/y.shape[0]
  loss+=0.5 * reg * np.sum(W * W)
  v_1=None
  v_2=None
  v_3=None
  v_4=None
  # from IPython.core.debugger import Tracer
  # Tracer()() #this one triggers the debugger
  
  ## TODO differneces of scores still has yi. This must be removed.
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
