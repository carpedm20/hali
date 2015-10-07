-- reference : https://github.com/facebook/SCRNNs/blob/master/tokenizer.lua
require 'utils'

local xlua = require 'xlua'
local pl = require('pl.import_into')()
local ffivector = require('fb.ffivector')

local Tokenizer = {}

local langs = config.langs
local new_line_symbol, unknown_symbol = '</s>', '</unk>'

--===================
-- Build Dictionary
--===================

function Tokenizer.make_dictionary(config, filename)
  local threshold = config.threshold or 0

  local dict = {}
  local w_count = {}
  for _, lang in pairs(langs) do
    local max_dict_size = 500000
    dict[lang] = {}
    dict[lang].symbol_to_index = {}
    dict[lang].index_to_symbol = {}
    dict[lang].index_to_frequency = torch.Tensor(max_dict_size)

    w_count[lang] = 1

    dict[lang].symbol_to_index[unknown_symbol] = w_count[lang]
    dict[lang].index_to_symbol[w_count[lang]] = unknown_symbol
    dict[lang].index_to_frequency[w_count[lang]] = 0
  end

  idx = nil
  line_counter = 1
  for line in io.lines(filename) do
    lang = langs[(line_counter+1) % 2 + 1]
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
    -- resize from max_dict_size to actual size
    dict[lang].index_to_frequency:resize(w_count[lang])
    print("# of total unique " .. lang .. " words : " ..
            w_count[lang] .. "/" .. dict[lang].index_to_frequency:sum())
  end

  -- Threshold
  total_word = {}
  removed = {}
  for _, lang in pairs(langs) do
  total_word[lang] = 1
  removed[lang] = 0
  for idx=2, w_count[lang] do
    word = dict[lang].index_to_symbol[idx]
    if dict[lang].index_to_frequency[idx] >= threshold then
      total_word[lang] = total_word[lang] + 1
      dict[lang].symbol_to_index[word] = total_word[lang]
      dict[lang].index_to_symbol[total_word[lang]] = word
      dict[lang].index_to_frequency[total_word[lang]] = dict[lang].index_to_frequency[idx]
    else
      dict[lang].symbol_to_index[word] = 1
      dict[lang].index_to_frequency[1] = dict[lang].index_to_frequency[1] + dict[lang].index_to_frequency[idx]
      dict[lang].index_to_frequency[idx] = 0
      removed[lang] = removed[lang] + 1
    end
  end
  dict[lang].index_to_frequency:resize(total_word[lang])

  print("# of total unique " .. lang .. " words : " .. total_word[lang] ..
        "/" .. dict[lang].index_to_frequency:sum() ..
        " (Removed " .. removed[lang] .. " " .. lang .. " words)")
  end

  -- Sorting
  for _, lang in pairs(langs) do
    -- true : descending order
    local sorted_frequency, sorted_index = torch.sort(dict[lang].index_to_frequency, true)
    sorted_frequency:div(math.max(1, dict[lang].index_to_frequency:sum()))

    dict_fname = paths.concat(config.data_path,
                              config.name .. '.dictionary' ..
                                '_lang=' .. lang ..
                                '_threshold=' .. config.threshold ..
                                '.th7')
    torch.save(dict_fname, dict[lang])
  end

  return dict
end

--===================
-- Make tokens
--===================

function Tokenizer.tokenize(dict, config, filenameIn, filenameOut)
  print("saving to " .. filenameOut)
  local unk = unknown_symbol
  local threshold = config.threshold
  local eos = config.eos
  local all_lines = ffivector.new_string()
  local tot_nr_words = 0
  local tot_lines = 0
  for s in io.lines(filenameIn) do
    tot_lines = tot_lines + 1
    all_lines[tot_lines] = s
    s = s:gsub("\t", "")
    s = s:gsub("^%s+", ""):gsub("%s+$", "")
    s = s:gsub("%s+", " ")
    local words = pl.utils.split(s, ' ')
    tot_nr_words = tot_nr_words + #words
    tot_nr_words = tot_nr_words + 1
  end
  print('-- total lines: ' .. tot_lines)
  local perm_vec
  if shuff == true then
    print('-- shuffling the data')
    perm_vec = torch.randperm(tot_lines)
  else
    print('-- not shuffling the data')
    perm_vec = torch.range(1, tot_lines)
  end
  local data = {}
  for _, lang in pairs(langs) do
    data[lang] = torch.Tensor(tot_nr_words)
  end
  local id = 0
  local cnt = 1
  for ln = 1, tot_lines do
    if ln % 100 == 0 then xlua.progress(ln, tot_lines) end

    if ln % 2 == 1 then
      lang = langs[1]
    else
      lang = langs[2]
    end
    local s = all_lines[perm_vec[ln]]
    s = s:gsub("\t", "")
    s = s:gsub("^%s+", ""):gsub("%s+$", "")
    s = s:gsub("%s+", " ")
    collectgarbage()
    local words = pl.utils.split(s, ' ')
    for i, word in pairs(words) do
      if word ~= "" then
        if dict[lang].symbol_to_index[word] == nil or
        dict[lang].index_to_frequency[dict[lang].symbol_to_index[word]] < threshold then
          print('WARNING: ' .. word .. ' being replaced by ' .. unk)
          id = dict[lang].symbol_to_index[unk]
        else
          id = dict[lang].symbol_to_index[word]
        end
        data[lang][cnt] = id
        cnt = cnt + 1
      end
    end
    if eos == true then
      id = dict[lang].symbol_to_index[new_line_symbol]
      data[lang][cnt] = id
      cnt = cnt + 1
    end
    collectgarbage()
  end
  torch.save(filenameOut, data)

  return data
end

return Tokenizer
