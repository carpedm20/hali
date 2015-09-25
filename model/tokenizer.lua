-- based on https://github.com/facebook/SCRNNs/blob/66f2325f9e003d2bfb7bf98967b948a1b14036bf/tokenizer.lua
require('math')
local ffivector = require('fb.ffivector')
local pl = require('pl.import')() -- Penlight Lua library : http://stevedonovan.github.io/Penlight/api/index.html

local Tokenizer = {}

function Tokenizer.build_dictionary(config, trainFileName)
  local kMaxDictSize = 500000
  local dict = {}
  dict.symbol_to_index = {} -- string to id
  dict.index_to_symbol = {} -- id to string

