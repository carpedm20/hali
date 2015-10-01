-- code highly referenced https://github.com/facebook/SCRNNs/blob/master/rnn.lua
require 'torch'
require 'sys'
require 'nn'

local RNN = torch.class("RNN")

-- Simple recurrent neural network
function RNN:__init(config, nets, criterion)
  -- config:
  --   n_hidden : # of hidden unites (size of the state)
  --   initial_val : value of the initial state
  --   backprop_freq : # of steps between two backprops and parameter updates
  --   backprop_len : # of backward step during each backprop (should be >= backprop_freq)
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
    self.nets.decoder = nets.decoder:clone()
    self.criterion = criterion:clone()
  else
    assert(nets.decoder_with_loss ~= nil)
    self.nets_decoder_with_losss = nets.decoder_with_loss:clone()
  end

  self.type = torch.Tensor():type()
  self.initial_val = config.initial_val
  self.initial_state_dim = config.initial_state_dim
  self.backprop_freq = config.backprop_freq
  self.batch_size = config.batch_size
  self.cuda_device = config.cuda_device
  if self.cuda_device then
    self:cuda()
  end

  self.unroll(config.backprop_len)
  self:recomputeParameters()
  self:reset()

  -- set clipping function
  local scale_clip = function(data, th)
    local data_norm = data:norm()
    if data_norm > th then
      data:div(data_norm/th)
    end
  end
  local hard_clip = function (vector, th)
    local tmp = vec:float()
    -- torch.data() : Returns a LuaJIT FFI pointer to the raw data of the tensor
    -- Accessing the raw data of a Tensor like this is extremely efficient.
    -- https://github.com/torch/torch7/blob/master/doc/tensor.md#result-datatensor-asnumber
    local tmpp = torch.data(tmp)
    for i = 0, tmp:size(1) - 1 do
      if tmpp[i] < -th then
        tmpp[i] = -th
      else
        if tmpp[i] > th then
          tmpp[i] = th
        end
      end
    end

    vec[{}] = tmp[{}] -- copy values to other Tensor
  end
  if config.clip_type == 'scale' then
    self.clip_function = scale_clip
  elseif config.clip_type == 'hard' then
    self.clip_function = hard_clip
  else
    error('Wrong clipping function')
  end

  self:set_internal_layers()
end


-- Reset network : reset training parameters and hidden state
function RNN:reset()
  self.i_input = 0 -- index for input
  self.dw:zero()
  self.state = nil
end


-- clone RNN and get RNN with same parameters but in different storages (memory)
function RNN:clone()
  -- torch.MemoryFile([mode])
  -- Constructor which returns a new MemoryFile object using mode. Valid mode are "r" (read), "w" (write) or "rw" (read-write). Default is "rw".
  local f = torch.MemoryFile("rw"):binary()
  f:writeObject(self)
  f:seek(1)
  local clone = f:readObject()
  f:close()
  return clone
end


-- [1]
function RNN:cuda()
  self.type = 'torch.CudaTensor'
  if self.criterion then
    self.criterion = self.criterion:cuda()
  else
    -- conflict with original code. https://github.com/facebook/SCRNNs/blob/master/rnn.lua#L122
    -- need debugging to find which is correct..
    self.nets.decoder_with_loss = self.nets.decoder_with_loss:cuda()
  end
  if self.unrolled_nets then
    for i=1,#self.unrolled_nets do
      for _, v in pairs(self.unrolled_nets[i]) do
        v = v:cuda()
      end
    end
  end
end


-- [2]
-- What is the meaning of unroll ???
function RNN:unroll(n)
  self.unrolled_nets = {}
  for i=1, n do
    self.unrolled_nets[i] = {}
    self.unrolled_nets[i].decoder_gradInput = torch.Tensor():type(self.type)

    for k,v in pairs(self.nets) do
      if (k ~= 'decoder') and (k ~= 'decoder_with_loss') then -- which are series of enc and dec
        self.unrolled_nets[i][k] = v:clone("weight", "gradWeight", "bias", "gradBias")
      end
    end
  end
end


-- [3]
function RNN:recomputeParameters()
  local dummy = nn.Sequential()
  for k, v in paris(self.nets) do
    if torch.typename(v) ~= 'nn.HSM' then
      dummy:add(v)
    end
  end
  self.w, self.dw = dummy:getParameters()
  self.mom = torch.Tensor(self.dw:size()):zero():type(self.type)
  self:unroll(#self.unrolled_nets)
end


-- [4]
-- Just save the initial layer pointer?
function RNN:set_internal_layers()
  self.ilayers = {}
  self.ilayers.embed = self.net.encoder.modules[1].modules[1] -- ???
  self.ilayers.project = self.net.encoder.modules[1].modules[2] -- ???
end


-- return a tensor filled with initial state
function RNN:get_initial_state(bsize)
  -- bsize : batch size
  local initial_state
  if self.initial_state_dim ~= nil then
    initial_sate = torch.Tensor(torch.LongStorage(self.initial_state_dim)):type(self.type)
  else
    initial_sate = torch.Tensor(bsize, self.n_hidden):type(self.type)
  end
  initial_sate(self.initial_val)
  return initial_state
end


-- Run forward pass with previous state prev_state
function RNN:elemForward(nets, input, prev_state, target)
  local bsize = input:size(1)
  prev_state = prev_state or self:get_initial_state(bsize)

  --store the local inputs and previous state
  nets.input = input
  nets.prev_state = prev_state

  local out_encoder = nets.encoder:forward{input, prev_state}
  local out_decoder, err, n_valid = nil, nil, nil
  if self.nets.decoder ~= nil then
    assert(self.nets.decoder_with_loss == nil)
    out_decoder = self.nets.decoder(out_encoder)
    if target then
      -- train out_decoder
      err, n_valid = self.criterion:forward(out_decoder, target)
    end
  else
    assert(self.nets.decoder_with_loss ~= nil)
    -- why here they don't check `target` like upper case
    err, n_valid = self.nets.decoder_with_loss:forward(out_decoder, target)
  end
  n_valid = n_valid or input:size(1)
  return out_decoder, out_encoder, err, n_valid
end


-- Run backward pass on the decode+criterion
function RNN:elemDecodeBackward(nets, target, learning_rate)
  if self.nets.decoder ~= nil then
    assert(self.nets.decoder_with_loss == nil)
    local decoder_output = self.nets.decoder.output
    local derr_do = self.criterion:backward(decoder_output, target)
    local gradInput = self.nets.decoder:backward(nets.encoder.output, derr_do)

    nets.decoder_gradInput:resizeAs(gradInput):copy(gradInput)
  else
    assert(self.nets.decoder_with_loss ~= nil)
    local gradInput
