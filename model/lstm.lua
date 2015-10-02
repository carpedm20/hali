require('torch')
require('sys')
require('nn')
require('rnn')

local LSTM = torch.class("LSTM", "RNN")

-- Long Short Term Memory
function LSTM:__init(config, nets, criterion)
  -- config:
  --   n_hidden : # of hidden unites (size of the state)
  --   initial_val : value of the initial state
  --   backprop_freq : # of steps between two backprops and parameter updates
  --   backprop_len : # of backward step during each backprop (should be >= backprop_freq)
  -- nets : network model from `model_factory.lua`
  --   encoder : produces h_t using x_t and h_{t-1}
  --   decoder : transformation applied to h_t to produce output vector (the next symbol)
  --                                   y1                          y2
  --                                    ^                           ^
  --                                  decoder                     decoder
  --                                    ^                           ^
  -- ... {h0, c0} -> lstm_encoder -> {h1, c1} -> lstm_encoder -> {h2, c2} ->
  --                      ^                           ^
  --                     x1                          x2
  self.n_hidden = config.n_hidden
  self.nets = {encoder = nets.encoder:clone()}
  if nets.decoder ~= nil then
    self.nets.decoder = nets.decoder:clone()
    self.criterion = criterion:clone()
  else
    assert(nets.decoder_with_loss ~= nil)
    self.nets_decoder_with_loss = nets.decoder_with_loss:clone()
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
    error('wrong clip type: ' .. config.clip_type)
  end

  self:set_internal_layers(ilayers)
end

function LSTM:set_internal_layers(layers)
  self.ilayers = {}
  for name, node in pairs(layers) do
    local id = node.id
    self.ilayers[name] = self.nets.encoder.fg.nodes[id].data.module
  end
end

function LSTM:unroll(n)
  self.unrolled_nets = {}
  -- params, gradParams : original parameters (memory)
  local params, gradParams = self.nets.encoder:parameters()
  local mem = torch.MemoryFile('w'):binary() -- creates a file in memory
  mem:writeObject(self.nets.encoder) -- writes the object into file (copy encoder)

  -- clone한 encoder의 storage를 기존의 encoder의 params, gradParams를 보게(view) 한다.
  -- 그냥 lstm에서 모든 t의 파라미터를 같게 하기 위해서 하는 방법.
  -- decoder는 복사를 안하는거 같은데 다른 코드에서 뭔가를 봤는데 기억이 안남..
  -- create `n` copy of LSTMs
  -- variable of interest : decoder_gradInput (no initialization), params, gradParams (copy)
  for i=1, n do
    self.unrolled_nets[i] = {}
    -- 여기 복사 안해도 되는건가?
    -- ex) nets.decoder_gradInput:resizeAs(gradInput):copy(gradInput)
    self.unrolled_nets[i].decoder_gradInput = torch.Tensor():type(self.type)
    local reader = torch.MemoryFile(mem:storage(), 'r'):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for j=1, #params do
      -- https://github.com/torch/torch7/blob/master/doc/tensor.md#self-setstorage-storageoffset-sizes-strides
      -- :set(storage)
      -- Tensor "view" the given storage. Any modification in the elements 
      -- of the Storage will have a impact on the elements of the Tensor.
      cloneParams[j]:set(params[j])
      cloneGradParams[j]:set(gradParams[j])
    end
    self.unrolled_nets[i]['encoder'] = clone
    collectgarbage()
  end
  mem:close()
end

function LSTM:get_initial_state(bsize)
  if not self.initial_state then
    self.initial_state = {}
    -- [1] : batch_size x n_hidden
    self.initial_state[1] = torch.Tensor(bsize, self.n_hidden):type(self.type)
    self.initial_state[2] = torch.Tensor(bsize, self.n_hidden):type(self.type)
    -- why two???
    self.initial_state[1]:fill(self.initial_val)
    self.initial_state[2]:fill(self.initial_val)
  end
  return self.initial_state
end


-- Run forward pass in the set of nets, with previous state, prev_state
function LSTM:elemForward(nets, input, prev_state, target)
  local bsize = input:size(1)
  prev_state = prev_state or self:get_initial_state(bsize)

  nets.input = input
  nets.prev_state = prev_state

  local out_encoder = nets.encoder:forward{input, prev_state}
  local out_decoder, err, n_valid = nil, nil, nil

  -- using the main net (not unrolled), which means using self.nets not nets.
  if self.nets.decoder ~= nil then
    assert(self.nets.decoder_with_loss == nil)
    out_decoder = self.nets.decoder:forward(out_decoder[1])
    if target then
      err, n_valid = self.criterion:forward(out_decoder, target)
    end
  else
    assert(self.nets.decoder_With_loss ~= nil)
    err, n_valid = self.nets.decoder_with_loss:forward(out_encoder[1], target)
  end
  n_valid = n_valid or input:size()
  return out_decoder, out_encoder, err, n_valid
end


-- run backward pass on the decode+criterion (or decode_with_loss) modules
function LSTM:elemDecoderBackward(nets, target, leraning_rate)
  if self.nets.decoder ~= nil then
    assert(self.nets.decoder_with_loss == nil)
    -- output = net:forward(input)
    -- criterion:forward(output, target)
    -- gradients = criterion:backward(output, target)
    -- gradInput = net:backward(input, gradients)
    local decoder_output = self.nets.decoder.output
    local derr_do = self.criterion:backward(decoder_output, target)
    local gradInput = self.nets.decoder(nets.encoder.output[1], derr_do)

    nets.decoder_gradInput:resize(gradInput):copy(gradInput)
  else
    assert(self.nets.decoder_with_loss ~= nil)

    -- https://github.com/torch/nn/blob/master/doc/module.md#gradinput-backwardinput-gradoutput
    -- backpropagation step consist in computing two kind of gradients at input given gradOutput (gradients with respect to the output of the module)
    --
    -- http://facebook.github.io/fbnn/fbnn/
    -- fbnn.HSM:updateGradInput(input, target)
    ---- Note: call this function at most once after each call updateOutput
    -- fbnn.HSM:accGradParameters(input, target, scale, direct_update)
    ---- scale must be set to the negative learning rate (-learning_rate)
    ---- Before calling this function you have to call HSM:updateGradInput first.
    local gradInput = self.nets.decoder_with_loss:updateGradInput(nets.encoder.output[1], target)
    nets.decoder_gradInput:resizeAs(gradInput):copy(gradInput)

    assert(torch.typename(self.nets.decoder_with_loss) == 'nn.HSM')
    self.nets.decoder_with_loss:accGradParameters(
      nets.encoder.output[1], target, -learning_rate, true)
  end
end

-- main training function
---- input : input word or minibatch
---- label : target word or minibatch
function LSTM:netInputTrain(input, label, params)
  self.i_input = self.i_input + 1
  local last_nets = self.unrolled_nets[1]
  for i = 1, #self.unrolled_nets - 1 do
    self.unrolled_nets[i] = self.unrolled_nets[i + 1]
  end

