import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  next_h_affine = x.dot(Wx) + prev_h.dot(Wh)+ b
  next_h = np.tanh(next_h_affine)
  cache = (x, prev_h, Wx, Wh, b, next_h,next_h_affine)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache

#1
def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (x, prev_h, Wx, Wh, b, next_h,next_h_affine) = cache
  
  # x.dot(Wx) + prev_h.dot(Wh)+ b
  dtanh = 1-next_h**2
  dtanh = (dnext_h*dtanh)
  dx = dtanh.dot(Wx.T)
  dprev_h = dtanh.dot(Wh.T)
  dWx = x.T.dot(dtanh)
  dWh = prev_h.T.dot(dtanh)
  db = np.sum(dtanh,axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db

      
def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  #(x, h0, Wx, Wh, b)
  # (x, prev_h, Wx, Wh, b, next_h) in cache
  n,t,d = x.shape
  h = Wh.shape[0]
  caches = []
  h = np.zeros((n,t,h))
  next_h,cache0 = rnn_step_forward(x[:,0,:], h0, Wx, Wh, b) #N, 0, H
  caches.append(cache0)
  h[:,0,:] = next_h
  if t>1:
    for time_step in range(1,t):
      next_h,cache = rnn_step_forward(x[:,time_step,:], next_h, Wx, Wh, b)
      h[:,time_step,:] = next_h
      caches.append(cache)
    
    cache = caches
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  # (x, prev_h, Wx, Wh, b, next_h) in cache
  (x, prev_h, Wx, Wh, b, next_h,next_h_affine) = cache[0]
  n,d = x.shape
  t = len(cache)
  h = Wh.shape[0]
  dx,dh0,dWx,dWh,db = np.zeros((n,t,d)), np.zeros((n,h)), np.zeros(Wx.shape), np.zeros(Wh.shape),np.zeros(b.shape)

  if len(cache)>-1:
    for i in np.arange(len(cache)-1,-1,-1):
      c = cache[i]
      (x, prev_h, Wx, Wh, b, next_h,next_h_affine) = c
      (dx_i, dprev_h_i, dWx_i, dWh_i, db_i) = rnn_step_backward(dh[:,i,:]+dh0, c)
      dx[:,i,:]+=dx_i
      #TODO research dh[:,i,:]+dh0 and dh0=dprev_h_i
      dh0 = dprev_h_i
      dtanh = 1-next_h**2
      dtanh = (dh[:,i,:]*dtanh)
      dWx += dWx_i
      dWh +=dWh_i
      db +=db_i
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  out = W[x]
  cache = (x,W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  (x,W) = cache
  dW = np.zeros(W.shape)
  np.add.at(dW, x,dout)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  H = prev_c.shape[1]
  a = x.dot(Wx) + prev_h.dot(Wh) + b
  ai,af,ao,ag = a[:,:H],a[:,H:2*H],a[:,(2*H):3*H],a[:,(3*H):]
  
  i = sigmoid(ai)
  f = sigmoid(af)
  o = sigmoid(ao)
  g = np.tanh(ag)

  next_c = f*prev_c + i*g
  next_h = o*np.tanh(next_c)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  cache = (prev_c,next_c,prev_h,next_h,x, Wx, Wh, b,g,o,f,i,a,ai,af,ao,ag,H)
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  (prev_c,next_c,prev_h,next_h,x, Wx, Wh, b,g,o,f,i,a,ai,af,ao,ag,H) = cache
  # first layer derivatives
  dai_dx = Wx[:,:H] # D,4H
  daf_dx = Wx[:,H:2*H]
  dao_dx = Wx[:,2*H:3*H]
  dag_dx = Wx[:,3*H:]
  da_dWx = x # N,D
  
  dai_dh_prev = Wh[:,:H] #H,4H
  daf_dh_prev = Wh[:,H:2*H]
  dao_dh_prev = Wh[:,2*H:3*H]
  dag_dh_prev = Wh[:,3*H:]
  da_dWh = prev_h # N,H
  
  # second layer derivatives
  di = sigmoid(ai)*(1-sigmoid(ai))
  df = sigmoid(af)*(1-sigmoid(af))
  do = sigmoid(ao)*(1-sigmoid(ao))
  dg = 1-np.tanh(ag)**2
  
  # third layer derivatives
  dnext_ci = di * g  
  dnext_cf = df * prev_c
  dnext_cg = i * dg
  dnext_c_total = dnext_ci + dnext_cf + dnext_cg
  
  # fourth layer derivatives
  dh_o =  do * np.tanh(next_c)
  dh_c =  o * (1-np.tanh(next_c)**2)
  
  # derivatives with respect to loss function
  dl_ho = dnext_h * dh_o
  dlh_c = dnext_h * dh_c
  
  dl_c = dnext_c * dnext_c 
  dl_ci = dnext_c * dnext_ci
  dl_cf = dnext_c * dnext_cf
  dl_cg = dnext_c * dnext_cg
  
  dl_hi = dlh_c * dnext_ci
  dl_hf = dlh_c * dnext_cf
  dl_hg = dlh_c * dnext_cg
  ############################################################
  # final derivatives with respect to loss # order - i,f,o,g #
  ############################################################
  # db 
  db = np.zeros(4*H)
  db_ci = np.sum(dl_ci,axis=0)
  db_cf = np.sum(dl_cf,axis=0)
  db_cg = np.sum(dl_cg,axis=0)
  
  db_hi = np.sum(dl_hi,axis=0)
  db_hf = np.sum(dl_hf,axis=0)
  db_ho = np.sum(dl_ho,axis=0)
  db_hg = np.sum(dl_hg,axis=0)
  
  db = np.concatenate((db_ci+db_hi,db_cf+db_hf,db_ho,db_cg+db_hg ), axis=0)
  # dx
  dl_xi = (dl_hi).dot(dai_dx.T) + dl_ci.dot(dai_dx.T) 
  dl_xf = (dl_hf).dot(daf_dx.T) + dl_cf.dot(daf_dx.T) 
  dl_xo =  dl_ho.dot(dao_dx.T) 
  dl_xg = (dl_hg).dot(dag_dx.T) + dl_cg.dot(dag_dx.T) 
  dx = dl_xi + dl_xf + dl_xo +dl_xg
  
  # dprev_h
  dl_hi = (dl_hi).dot(dai_dh_prev.T) + dl_ci.dot(dai_dh_prev.T) 
  dl_hf = (dl_hf).dot(daf_dh_prev.T) + dl_cf.dot(daf_dh_prev.T) 
  dl_ho =  dl_ho.dot(dao_dh_prev.T) 
  dl_hg = (dl_hg).dot(dag_dh_prev.T) + dl_cg.dot(dag_dh_prev.T) 
  dprev_h = dl_hi + dl_hf + dl_ho +dl_hg
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

