import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.
  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.
    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    w1 = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim,hidden_dim))
    self.params['W1'] = w1
    self.params['b1'] = np.zeros(hidden_dim)

    w2 = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim,num_classes))
    self.params['W2'] = w2
    self.params['b2'] = np.zeros(num_classes)
    # self.params['b1'] = np.ones()
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.
    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    L1, cache1 = affine_forward(X,self.params['W1'],self.params['b1'])
    L2, cache2 = affine_forward(L1,self.params['W2'],self.params['b2'])

    scores = L2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    loss, dloss = softmax_loss(L2,y)  
    reg_loss = 0.5*self.reg*(np.sum(self.params['W2']*self.params['W2'])+np.sum(self.params['W1']*self.params['W1']))
    loss += reg_loss
    dL2, dW2, db2 = affine_backward(dloss,cache2)
    
    dW2+=self.reg*self.params['W2']

    grads['W2'] = dW2
    grads['b2'] = db2

    dL, dW1, db1 = affine_backward(dL2,cache1)

    dW1+=self.reg*self.params['W1']
    grads['W1'] = dW1
    grads['b1'] = db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    for i in range(0,self.num_layers):
      wieght = "W%s"%( i+1 )
      bias = "b%s"%( i+1 )
      
      if i < self.num_layers-1:
        hidden_dim = hidden_dims[i]
      
      if i > 0 and i< self.num_layers-1:
        previous_layer = hidden_dims[i-1]
      elif i == self.num_layers-1:
        previous_layer = hidden_dims[i-1]
        
      if  (i< self.num_layers-1) & self.use_batchnorm:
          gamma = "gamma%s"%( i+1 )
          beta = "beta%s"%( i+1 )
          self.params[gamma] = np.ones(hidden_dim)
          self.params[beta] = np.zeros(hidden_dim)
      if i == 0:
        self.params[bias] = np.zeros(hidden_dim)
        self.params[wieght] = np.random.normal(loc=0.0, 
                                  scale=weight_scale, 
                                  size=(input_dim,hidden_dim))
      else:
        # last layer
        if i == self.num_layers-1:
          self.params[bias] = np.zeros(num_classes)
          self.params[wieght] = np.random.normal(loc=0.0, 
                                  scale=weight_scale, 
                                  size=(previous_layer,num_classes))
        else:
          self.params[bias] = np.zeros(hidden_dim)
          self.params[wieght] = np.random.normal(loc=0.0, 
                                    scale=weight_scale, 
                                    size=(previous_layer,hidden_dim))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
  ## TODO
  # CREATE batch helper function
  def batch_affine_relu_forward(self,x,w,b, gamma, beta,bn_param):
    a_out, fc_cache = affine_forward(x, w, b)
    b_out, b_cache = batchnorm_forward(a_out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(b_out)
    cache = (fc_cache,b_cache ,relu_cache)
    return out, cache

  def batch_affine_relu_backward(self,dout,cache):
    fc_cache, b_cache,relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = batchnorm_backward(da,b_cache)
    dx, dw, db = affine_backward(dx, fc_cache)
    return dx, dw, db, dgamma, dbeta
    
  
  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    previous_layer=None
    all_cache_layers = []

    for layer in range(1,self.num_layers+1):
      wieght = "W%s"%( layer )
      bias = "b%s"%( layer )
      cache = "Cache%s"%(layer)
      gamma = 'gamma%s'%(layer)
      beta = 'beta%s'%(layer)
      layer_i = layer-1
      # my forward pass
      if layer == 1:
        if self.use_batchnorm:
          
          scores, result_cache = self.batch_affine_relu_forward(X,self.params[wieght],self.params[bias], self.params[gamma], self.params[beta],self.bn_params[layer_i])
        else:
          scores, result_cache = affine_relu_forward(X,self.params[wieght],self.params[bias])
        previous_layer = scores
      elif layer<self.num_layers:
        if self.use_batchnorm:
          ## TODO write batchnorm forward
          scores, result_cache = self.batch_affine_relu_forward(previous_layer,self.params[wieght],self.params[bias], self.params[gamma], self.params[beta],self.bn_params[layer_i])
          ### MADE THIS  CHANGE BUT IT WAS WORKING FINE BEFORE
          previous_layer = scores
        else:
          scores, result_cache = affine_relu_forward(previous_layer,self.params[wieght],self.params[bias])
          previous_layer = scores
      else:
        scores, result_cache = affine_forward(previous_layer,self.params[wieght],self.params[bias])
        previous_layer = scores
      if True and self.use_dropout and layer < self.num_layers:
        scores, cache = dropout_forward(previous_layer,self.dropout_param)
        previous_layer = scores
        
        result_cache = result_cache+cache
      all_cache_layers.append(result_cache)
      # self.params[layer] = layer
      
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores
    
    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dloss = softmax_loss(scores,y) 
    for layer in range(self.num_layers,0,-1):
      wieght = "W%s"%( layer )
      bias = "b%s"%( layer )
      reg_loss = 0.5*self.reg*(np.sum(self.params[wieght]*self.params[wieght]))
      loss += reg_loss
    
    previous_layer = None
    for layer in range(self.num_layers,0,-1):
      weight = "W%s"%( layer )
      bias = "b%s"%( layer )
      gamma = 'gamma%s'%(layer)
      beta = 'beta%s'%(layer)
      layer_i = layer-1
      cache = all_cache_layers[layer_i]
      if layer  == self.num_layers:
        dL, dW, db = affine_backward(dloss,cache)
        previous_layer = dL
      else:
        if True and self.use_dropout:
          # print all_cache_layers[0][-2]
          cache_list = list(cache)
          mask = cache_list.pop()
          dropout_param = cache_list.pop()
          dropout_cache = (dropout_param,mask)
          
          cache = tuple(cache_list)
          dL = dropout_backward(previous_layer,dropout_cache)
          previous_layer = dL
        if self.use_batchnorm:
          ## TODO write batchnorm forward
          dL, dW, db, dgamma, dbeta =  self.batch_affine_relu_backward(previous_layer,cache)
          previous_layer = dL
          grads[gamma] = dgamma
          grads[beta] = dbeta 
        else:
          dL, dW, db = affine_relu_backward(previous_layer,cache)
          previous_layer = dL
      dW+=self.reg*self.params[weight]
      # previous_layer = dL
      grads[weight] = dW
      grads[bias] = db


    # dL2, dW2, db2 = affine_backward(dloss,cache2)
    
    # dW2+=self.reg*self.params['W2']

    # grads['W2'] = dW2
    # grads['b2'] = db2

    # dL, dW1, db1 = affine_backward(dL2,cache1)

    # dW1+=self.reg*self.params['W1']
    # grads['W1'] = dW1
    # grads['b1'] = db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return loss, grads

    
