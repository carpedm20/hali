local langs = {'en','ko'}

local config = {
  langs = langs,
  filename = {train=string.format('%s_%s.train.txt',langs[1],langs[2]),
              test=string.format('%s_%s.test.txt',langs[1],langs[2]),
              valid=string.format('%s_%s.valid.txt',langs[1],langs[2])},
  token_filename = string.format('%s_%s.token.th7',langs[1],langs[2]),
  threshold = 0,
  batch_size = 3,
  data_path = './data',
  root= './',
  eos = true,
  name = string.format('%s_%s',langs[1],langs[2]),
  n_hidden = 100,
  n_layers = 4,
}

return config
