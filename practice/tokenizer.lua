-- based on https://github.com/facebook/SCRNNs/blob/66f2325f9e003d2bfb7bf98967b948a1b14036bf/tokenizer.lua
--
require('math')
local ffivector = require('fb.ffivector')
local pl = require('pl.import')() -- Penlight Lua library : http://stevedonovan.github.io/Penlight/api/index.html

local Tokenizer = {}

function Tokenizer.build_dictionary(config, trainFileName)
  local kMaxDictSize = 500000
  local dict = {}
  dict.symbol_to_index = {} -- string to id
  dict.index_to_symbol = {} -- id to string
  dict.index_to_freq = torch.Tensor(kMaxDictSize)
  dict.index_to_cluster = nil
  dict.index_to_index_within_cluster = nil
  dict.cluster_to_index = {}
  dict.mapping = nil
  
  local n_clusters = config.nclusters
  local threshold = config.threshold

  print("[*] loading: "..trainfname)
  local n_words = 1 -- # of unique words
  local tot_n_words = 0 -- total # of words in corpus
  local unk = "<UNK>"
  dict.symbol_to_index[unk] = n_words
  dict.index_to_symbol[n_words] = unk
  dict_index_to_freq[n_words] = 1
  local count = 0
  for s in io.lines(trainfname) do
    s = s:gsub("\t","") -- remove tabs
    s = s:gsub("%s+"," ") -- replace spaces into a single space
    local words = pl.utils.split(s, ' ') -- Penlight : return only words 
    -- pl.utils.split("asdf asdf asd", " ")
    -- {
    --   1 : "asdf"
    --   2 : "asdf"
    --   3 : "asd"
    -- }
    for i, word in pairs(words) do
      if word ~= "" then
        if dict.symbol_to_index[word] == nil then
          n_words = n_words + 1
          dict.symbol_to_index[word] = n_words
          dict.index_to_symbol[n_words] = word
          dict.index_to_freq[n_words] = 1
        else
          local index = dict.symbol_to_index[word]
          dict.index_to_freq[index] = dict.index_to_freq[index] + 1
        end
        count = count + 1
      end
    end

    -- Add \n for every line
    -- Why </s> ???
    if dict.symbol_to_index["</s>"] == nil then
      n_words = n_words + 1
      dict.symbol_to_index["</s>"] - n_words
      dict.index_to_symbol[n_words] = "</s>"
      dict.index_to_freq[n_words] = 1
    else
      local index = dict.symbol_to_index["</s>"]
      dict.index_to_freq[index] = dict.index_to_freq[index] + 1
    end
    count = count + 1
  end
  dict.index_to_freq:resize(n_words)
  tot_n_words = dict.index_to_freq:sum()
  print("[Done making the dictionary. There are " .. nr_words - 1 ..
         " unique words and a total of " .. tot_nr_words - 1 ..
         " words in the training set.]")

  -- map rare words to special token and skip indices
  -- if the specified threshold is greater than 0
  local removed = 0
  local net_nwords = 1
  if threshold > 0 then
    for i = 2, dict.index_to_freq:size(1) do
      local word = dict.index_to_symbol[i]
      if dict.index_to_freq[i] < threshold then
        -- dict.index_to_freq[1] == <UNK>
        dict.index_to_freq[1] = dict.index_to_freq[1] + dict.index_to_freq[i]
        dict.index_to_freq[i] = 0
        dict.symbol_to_index[word] = 1
        removed = removed + 1
      else
        -- re-adjust the indices to make them continuous
        net_nwords = net_nwords + 1
        dict.index_to_freq[net_nwords] = dict.index_to_freq[i]
        dict.symbol_to_index[word] = net_nwords
        dict.index_to_symbol[net_nwords] = word
      end
    end
    print('[Removed ' .. removed .. ' rare words. ' ..
          'Effective number of words ' .. net_nwords .. ']')
    dict.index_to_freq:resize(net_nwords)
  else
    -- no threshold mechanism..
    net_nwords = n_words
  end

  -- create the cluster index tensors
  dict.index_to_cluster = torch.LongTensor(net_nwords):fill(0)
  dict.index_to_index_within_cluster = torch.LongTensor(net_nwords):fill(0)

  -- [[ sort the tokens by frequency ]]

  local sorted_freqs, sorted_index = torch.sort(dict.index_to_freq, true)
  -- [res] torch.div([res,] tensor, value)
  -- Divide all elements in the tensor by the given value.
  -- ** `use tot_n_words` instead `net_nwords` **
  sorted_freqs:div(math.max(1, tot_n_words)) -- normalize
  if n_clusters == 0 then
    n_clusters = math.floor(math.sqrt(net_nwords))
  end

  local probab_mass = 1.0 / n_clusters
  local current_mass = 0
  local cluster_id = 1
  local within_cluaster_index = 0 

  for w=1, net_nwords do
    if current_mass < probab_mass then
      current_mass = current_mass + sorted_freqs[w]
      within_cluster_index = within_cluster_index + 1
    else
      -- move to next cluster
      cluster_id = cluster_id + 1
      current_mass = sorted_freqs[w]
      within_cluster_index = 1
    end

    dict.index_to_cluster[sorted_index[w]] = cluster_id
    dict.index_to_index_within_cluster[sorted_index[w]] = within_cluster_index
  end
  print("[Created #" .. cluster_id .. " cluster.]")

  -- Count how many words per cluster. index -> cluster_id
  local wordsPerCluster = torch.zeros(cluster_id)
  for w=1, net_nwords do
    local cur_cluster = dict.index_to_cluster[w]
    wordsPerCluseter[cur_cluster] = wordsPerCluster[cur_cluster] + 1
  end

  -- build reverse index from cluster id back to index. cluster_id -> index
  dict.mapping = torch.LongTensor(net_nwords, 2)
  for c=1, cluster_id do
    table.insert(dict.cluster_to_index, torch.LongTensor(wordsPerCluster[c]))
  end
  for w=1, net_nwords do
    local cur_cluster = dict.index_to_cluster[w]
    local cur_word = dict.index_to_index_within_cluster[w]

    dict.cluster_to_index[cur_cluster][cur_word] = w
    dict.mapping[w][1] = cur_cluster
    dict.mapping[w][2] = cur_word
  end
  dict.seperatorIndex = dict.symbol_to_index['</s>']

  -- save directory.
  local dict_fname = paths.concat(config.dest_path,
                                  config.name .. '.directory' ..
                                  '_ncluster=' .. n_clusters ..
                                  '_thresh=' .. config.threshold ..
                                  '.th7')
  torch.save(dict_fname, dict)
  print('There are ' .. net_words .. ' words in the corpus.')

  return dict, n_clusters
end
