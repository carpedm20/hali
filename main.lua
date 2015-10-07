-- require('mobdebug').start()
debugger = require 'fb.debugger'

require 'nn' 
require 'paths'
require 'nngraph' 
require 'optim'
require 'fbcunn' 

LSTM = require 'model.LSTM'
config = require 'config'

local tokenizer = require 'tokenizer'

dict = {}
data = {}

for _, ftype in ipairs({'train'}) do
  print("Making dictionary for " .. ftype .. "..")

  filename = paths.concat(config.data_path, config.filename[ftype])
  out_filename = paths.concat(config.data_path, config.filename[ftype]..'.data')

  dict[ftype] = {}
  for _, lang in pairs(config.langs) do
    dict_fname = paths.concat(config.data_path,
                              config.name .. '.dictionary' ..
                                '_lang=' .. lang ..
                                '_threshold=' .. config.threshold ..
                                '.th7')
    dict[ftype][lang] = torch.load(dict_fname)
  end
  data[ftype] = torch.load(out_filename)

  -- dict[ftype] = tokenizer.make_dictionary(config, filename)
  -- data[ftype] = tokenizer.tokenize(dict[ftype], config, filename, out_filename)
  print("End loading dictionary and data!")
end

rnn = LSTM.lstm(config, #dict.train.en.index_to_symbol)
