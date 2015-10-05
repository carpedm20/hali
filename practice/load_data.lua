require 'torch'
require 'paths'
require 'math'
require 'xlua'

local WordDataset = torch.class('WordDataset')

function WordDataset:__init(config)
  self.bsize = config.batch_size
  self.root = config.root_path
  self.sets = {}
  if config.train_file then
    self.train_file = paths.concat(self.root, config.train_file)
    self.sets['train'] = torch.load(self.train_file)
  end
  if config.valid_file then
    self.valid_file = paths.concat(self.root, config.valid_file)
    self.sets['valid'] = torch.load(self.valid_file)
  end
  if config.test_file then
    self.test_file = paths.concat(self.root, config.test_file)
    self.sets['test'] = torch.load(self.test_file)
  end
  collectgarbage()
end

function WordDataset:get_set_from_name(name)
  local out=self.sets[set_name]
  if out == nil then
    if name == 'nil' then
      error('set name is nil')
    else
      error('Unknown name: '.. name)
    end
  end
  return out
end

-- Return the data corresponding to train,valid,test sets
-- `sname` : the name of the data type. Return 2D tensor of size
-- (N/batch_size)*batch_size, where N is # of words.
function WordDataset:get_shard(sname)
  local set = self:get_set_from_name(sname)
  local shard_length = torch.floor(set:size(1)/self.batch_size)
  local cur_shard = torch.LongTensor(shard_length, self.batch_size)
  local offset = 1
  for i=1, self.batch_size do
    cur_shard[{{},i}]:copy(set[{{offset, offset+shard_length-1}}])
    offset = offset + shard_length
  end
  collectgarbage()
  collectgarbage()
  return cur_shard
end
