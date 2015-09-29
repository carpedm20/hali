-- code highly referenced https://github.com/facebook/SCRNNs/blob/master/rnn.lua
require 'torch'
require 'sys'
require 'nn'

local RNN = torch.class("RNN")

function RNN:__init(config, nets, criterion)
  -- config:
  --   n_hidden : # of hidden unites (size of the state)
  --   initial_val : value of the initial state
  --   backprop_freq : # of steps between two backprops and parameter updates
  --   backprop_len
  -- nets : network model from `model_factory.lua`
  --   encoder : produces h_t using x_t and h_{t-1}
  --   decoder : transformation applied to h_t to produce output vector (the next symbol)
  --              y1               y2              y3
  --              ^                ^               ^
  --           decoder          decoder         decoder
  --              ^                ^               ^
  -- ... h0 -> encoder -> h1 -> encoder -> h2 -> encoder -> h3
  --              ^                ^               ^
  --              x1               x2              x3
  self.n_hidden = config.n_hidden
  self.nets = {encoder = nets.encoder:clone()}
  if nets.decoder ~= nil then
