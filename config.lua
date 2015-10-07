local langs = {'en','ko'}

local config = {
  langs = langs,
  filename = string.format('%s_%s.txt',langs[1],langs[2]),
  token_filename = string.format('%s_%s.token.th7',langs[1],langs[2]),
  threshold = 4000,
  dest_path = './data',
  eos = true,
  name = string.format('%s_%s',langs[1],langs[2])
}

return config
