require 'utils'

pl = require('pl.import_into')()
config = require 'config'

langs = config.langs
filename = config.filename
n_cluster = config.n_cluster
threshold = config.threshold or 0

local new_line_symbol, unknown_symbol = '</s>', '</unk>'

dict = {}
w_count = {}
for _, lang in pairs(langs) do
  dict[lang] = {}
  dict[lang].symbol_to_index = {}
  dict[lang].index_to_symbol = {}
  dict[lang].index_to_frequency = {}
  dict[lang].index_to_cluster = {}

  w_count[lang] = 1

  dict[lang].symbol_to_index[unknown_symbol] = w_count[lang]
  dict[lang].index_to_symbol[w_count[lang]] = unknown_symbol
  dict[lang].index_to_frequency[w_count[lang]] = 0
end

if not file_exists(filename) then
  dofile('genearte.lua')
end

idx = nil
line_counter = 1
for line in io.lines(filename) do
  lang = langs[line_counter % 2 + 1]
  line_counter = line_counter + 1

  line = line:gsub("\t","")
  line = line:gsub("%s+"," ")

  local words = pl.utils.split(line, ' ')
  for i, word in pairs(words) do
    if word ~= "" then
      if dict[lang].symbol_to_index[word] == nil then
        w_count[lang] = w_count[lang] + 1
        dict[lang].symbol_to_index[word] = w_count[lang]
        dict[lang].index_to_symbol[w_count[lang]] = word
        dict[lang].index_to_frequency[w_count[lang]] = 1
      else
        idx = dict[lang].symbol_to_index[word]
        dict[lang].index_to_frequency[idx] = dict[lang].index_to_frequency[idx] + 1
      end
    end
  end
  
  if dict[lang].symbol_to_index[new_line_symbol] == nil then
    w_count[lang] = w_count[lang] + 1
    dict[lang].symbol_to_index[new_line_symbol] = w_count[lang]
    dict[lang].index_to_symbol[w_count[lang]] = new_line_symbol
    dict[lang].index_to_frequency[w_count[lang]] = 1
  else
    idx = dict[lang].symbol_to_index[new_line_symbol]
    dict[lang].index_to_frequency[idx] = dict[lang].index_to_frequency[idx] + 1
  end
end

for i, lang in pairs(langs) do
  print("# of total unique " .. lang .. " words : " .. w_count[lang])
end

print(#dict['en'].index_to_frequency, dict['en'].index_to_frequency)

-- Threshold
for _, lang in pairs(langs) do
  keep_count = 1
  for idx=2, w_count[lang] do
    if dict[lang].index_to_frequency[idx] >= threshold then
      keep_count = keep_count + 1
      word = dict[lang].index_to_symbol[idx]
      dict[lang].symbol_to_index[word] = keep_count
      dict[lang].index_to_symbol[keep_count] = word
      dict[lang].index_to_frequency[keep_count] = dict[lang].index_to_frequency[idx]
    else
      word = dict[lang].index_to_symbol[idx]
      dict[lang].symbol_to_index[word] = nil
      dict[lang].index_to_symbol[idx] = nil

      dict[lang].index_to_frequency[1] = dict[lang].index_to_frequency[1] + dict[lang].index_to_frequency[idx]
      dict[lang].index_to_frequency[idx] = nil
    end
  end
end

print(#dict['en'].index_to_frequency, dict['en'].index_to_frequency)
