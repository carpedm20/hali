require 'utils'

config = require 'config'

local langs = config.langs
local filename = config.filename

dict = {}
for _, lang in pairs(langs) do
  dict[lang] = {}
end

if not file_exists(filename) then
  dofile('genearte.lua')
end

local count = 0
for line in io.lines(filename) do
  local lang = langs[count % 2 + 1]
  -- print(line)

  line = line:gsub("\t","")
  line = line:gsub("%s+"," ")

  count += 1
end
