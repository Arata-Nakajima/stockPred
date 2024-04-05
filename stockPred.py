import reservoirpy as rpy
import matplotlib.pyplot as plt
import numpy as np
import csv
import random

from reservoirpy.datasets import narma
from reservoirpy.mat_gen import uniform, bernoulli
from reservoirpy.nodes import IPReservoir, LMS, FORCE
from scipy.stats import expon

#SCALE_FACTOR = 8 # for training
#OFFSET = 12800 # for training
SCALE_FACTOR = 3 # for prediction
OFFSET = 5835 # for prediction

# Data Load
filename = "stockPred_input.txt"

with open( filename, "r" ) as f:
  price_data = np.loadtxt( f, dtype = int )
  # if data format was ( days, price )
  # for line in f:
  #      parts = line.strip().split(',')
  #      x_data.append(int(parts[0]))
  #      y_data.append(int(parts[1]))

#plt.plot( x_data, y_data )
#plt.savefig( "price.png" )

price_data = np.reshape( price_data, ( len( price_data ), 1 ) )
print( 'data size', len( price_data ) )


# Config
rpy.verbosity(0)
rpy.set_seed(1)

#W = adjacency_matirx()
distrib_1 = ( 1 / max( price_data, key = abs ) ** 0.5 )
#print( distrib_1 )

#x_data = []
#y_data = []
window_size = 5 
train_ratio = 0.8

reservoir = IPReservoir(
    W = uniform( high = 1.0, low = -1.0),
    Win = uniform( high = distrib_1, low = -distrib_1 ),
    bias = uniform( high = distrib_1, low = -distrib_1 ),
    #internal_state = uniform( high = 1.0, low = -1.0 ),
    units = 100,
    #sr = 0.95,
    #mu = 0.3,
    learning_rate = 1e-3,
    #input_scaling = 1,
    rc_connectivity = 0.5,
    input_connectivity = 0.5,
    activation = "tanh",
    epochs = 100
)

reservoir.fit( price_data[:window_size], price_data[ window_size + 1 ] )
#print( reservoir.state() )
#print( reservoir.state().shape )

hidden_out = random.sample( reservoir.state().flatten().tolist(), int( 0.2 * 100 ))
#distrib_2 = ( 1 / min( hidden_out, key = abs ) ** 0.5 )
distrib_2 =  max( price_data, key = abs ) / SCALE_FACTOR 
print( "hout", distrib_2 )

readout = LMS( 
    Wout = uniform( high = distrib_2, low = -distrib_2 ),
    bias = uniform( high = distrib_2, low = -distrib_2 ),
    output_dim = 1,
)
esn_model = reservoir >> readout 

# Train ESN
train_len = round( len( price_data ) * train_ratio )
eval_len = len( price_data ) - train_len
train_data = price_data[ : train_len ]
#print( train_len, eval_len )

#esn_model = esn_model.fit( price_data[:len(price_data)-1], price_data[1:len(price_data)] )
#print( train_data.shape )
#print( train_data[ window_size + 1] )
#Y = train_data[ : window_size + 1 ]
#y = np.atleast_2d(Y[0, :])
#esn_model = esn_model.train( train_data[ :window_size ], train_data[ window_size + 1 ] )
#esn_model = esn_model.fit( train_data[ :window_size ], train_data[ window_size + 1 ] )
#esn_model = esn_model.train( train_data[:len(price_data)-1], train_data[1:len(price_data)] )
#for i in range ( 0, train_len ):
for i in range ( 0, 1000 ):
  window_data = price_data[ i : i + window_size ]
  #window_data = train_data[ i : i + window_size ]
  i_train = i + window_size
  #esn_model = esn_model.train( window_data, train_data[ i_train ] )
  #esn_model = esn_model.partial_fit( window_data, train_data[ i_train ] )
  reservoir = reservoir.partial_fit( window_data, price_data[ i_train ] )
  #print( i, i_train )
  #print( window_data, i )
  #print( train_data[ i_train ], i_train )

# Evaluate ESN
pred = np.zeros( len( price_data ) )
for i in range ( 0, window_size ):
  pred[i] = 10000
  print( i, pred[i], price_data[i], pred[i] - price_data[i] )
for i in range ( window_size, len( price_data ) + 1 ):
  #window_data = price_data[ i - window_size : i ]
  warmup_y = esn_model.run( price_data[ i - window_size : i ] )
  #warmup_y = esn_model.run( train_data[ i : i + window_size ], shift_fb = False )

#eval_data = price_data[ train_len : ]
#pred = np.zeros( len(eval_data) )
#for i in range ( 0, eval_len ):
#  warmup_y = esn_model.run( eval_data[ i : i + window_size ], shift_fb = False )
  #print( pred_next )
  #pred.append( pred_next )

  #Y_pred = np.empty((1, 1))
  x = warmup_y[-1].reshape(1, -1) + OFFSET
  #x = esn_model(x)
  pred[ i + 1 ] = x
  print( i, pred[i], price_data[i], pred[i] - price_data[i] )

#for i in range ( 0, len( price_data ) ):
#  print( i, pred[i], price_data[i] )

plt.plot( pred, linewidth = 0.5, color = "red" )
plt.plot( price_data, linewidth = 0.5, color = "green" )
#plt.plot( eval_data )
plt.savefig( "stockPred.png" )

#print( pred.shape, price_data.shape)
#np.savetxt( 'stockPred.csv', np.column_stack( ( pred, price_data )), delimiter = ',', fmt = "%s" )
#with open( 'stockPred.csv', 'w', newline = '' ) as csvfile :
#  writer = csv.writer( csvfile )
#  writer.writerows( zip( x, price_data ))
