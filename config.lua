local langs = {'en','ko'}

local config = {
  langs = langs,
  filename = string.format('%s_%s.txt',langs[1],langs[2])
}

return config
