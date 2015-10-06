-- require('mobdebug').start()
debugger = require 'fb.debugger'

require 'nn' 
require 'nngraph' 
require 'optim'
require 'fbcunn' 

LSTM = require 'model.LSTM'

config = {n_hidden=10, n_classes=2, n_layers=10}
rnn = LSTM.lstm(config)
debugger.enter()
