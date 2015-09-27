require 'torch'
require 'sys'
require 'nn'

local RNN = torch.class("RNN")

function RNN:__init(config, nets, criterion)
  self.n_hidden = config.n_hidden
  self.nets = {encoder = nets.encoder:clone()}
