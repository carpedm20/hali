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
end
