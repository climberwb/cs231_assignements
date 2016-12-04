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
  # scores = X.dot(W)
  # loss= -fyi +log(sum(e^fyj))
  # loss = -W.dotX +log(e^score1+e^score2+e^score3)
                    #score1 = w[:,1]*xi+w[:,2]*xi+w[:,2]*xi
  # loss/dw = -X + 1/e^score1+e^score2+e^score3)*[e^(w[:,1].X)*X  ]
  
  for i,xi in enumerate(X):
    label=y[i]
    f = []
    f_delta=[]
    for class_i,w in enumerate(W.T):
      f.append(xi.dot(w.T))
      f_delta.append(np.exp(xi.dot(w.T))*xi)
    # from IPython.core.debugger import Tracer
    # Tracer()() #this one triggers the debugger
    for class_i,w in enumerate(W.T):
      if class_i == label:
        dW[:,class_i][:,None] +=  -xi.T[:,None] + (1/sum(np.exp(f))*f_delta[class_i][:,None])
      else:                            
        dW[:,class_i][:,None] +=   (1/sum(np.exp(f)))*f_delta[class_i][:,None]
    # int("d")
        
    f-=np.max(f)
    label=y[i]
    yi_score = f[label]
    loss += -(np.log(np.exp(yi_score) / np.sum(np.exp(f)))) 
  # for i,f in enumerate(scores):
  #   f-=np.max(f)
  #   label=y[i]
  #   yi_score = f[label]
  #   loss += -(np.log(np.exp(yi_score) / np.sum(np.exp(f)))) 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= X.shape[0] 
  loss += 0.5*reg*np.sum(W*W)
  dW /=X.shape[0]
  dW +=reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

