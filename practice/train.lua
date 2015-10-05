require 'math'
require 'sys'
require 'os'
require 'torch'
require 'xlua'

local RNNTrainer = torch.class('RNNTrainer')

function RNNTrainer:__init(config, model, dataset)
  self.training_params = {learning_rate = config.initial_learning_rate,
                          gradient_clip = config.gradient_clip,
                          gradInput_clip = config.gradInput_clip,
                          momentum = config.momentum}

  self.learning_rate_shrink = config.learning_rate_shrink
  self.shrink_multiplier = config.shrink_factor
  self.trbatches = config.trbatches
  self.unk_index = config.unk_index
  self.progress_bar = not config.no_progress
  self.model = model
  self.dataset = dataset
  self.save_dir = config.save_dir
  self.use_valid = config.use_valid_set
  self.use_test = config.use_test_set
  self.type = torch.Tensor():type()
  self.anneal_type = config.shrink_type
  self.annealed = false
end

function RNNTrainer:cuda()
  self.type = 'torch.CudaTensor'
end

function RNNTrainer:run_epoch_train()
  local total_err = 0
  local n_total = 0
  local n_words = 0
  self.model:reset()
  local shard = self.dataset:get_shard('train')
  if self.type == 'torchCudaTensor' then
    shard = shard:type(self.type)
  end
  local inputs = shard[{{1, shard:size(1)-1}}]
  local labels = shard[{{2, shard:size(1)}}]
  local batch_size = shard:size(2)
  n_words = n_words + inputs:size(1) * inputs:size(2)

