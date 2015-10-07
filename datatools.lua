--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Sumit Chopra <spchopra@fb.com>
--          Michael Mathieu <myrhev@fb.com>
--          Marc'Aurelio Ranzato <ranzato@fb.com>
--          Tomas Mikolov <tmikolov@fb.com>
--          Armand Joulin <ajoulin@fb.com>

-- this file has a collection of utilities used for data handling
require 'torch'
require 'paths'
require 'math'
require 'xlua'
require 'nn'
local tokenizer = require('tokenizer')
require('datasource')

local dtools = {}

-- generates the base torch data files
function dtools.generate_data(config)
    config_dataset.dest_path = config.name

    -- fetch the data if its not already fetched
    if not paths.dirp(config_dataset.dest_path) then
        print('[[ Data not found: fetching a fresh copy ]]')
        os.execute('mkdir -p ' .. config_dataset.dest_path)
        os.execute('th generate.lua')
    end

    -- check if the dictionary already exist or not
    local isdict = false
    local dictname
    for f in paths.files(config_dataset.root_path) do
        if string.find(f, 'dictionary') then
            isdict = true
            dictname = f
        end
    end
    -- make the dictionary if it does not already exists
    local dict
    if isdict == false then
        print('')
        print('[[ building dictionary ]]')
        local fname = config_dataset.filename['train']
        local inpath  = paths.concat(config_dataset.root_path, fname)
        dict, _ = tokenizer.make_dictionary(config, inpath)
    else
        dict = torch.load(paths.concat(config_dataset.root_path, dictname))
    end

    -- generate the training data if required
    local   fname = config_dataset.train_file
    local  inpath = paths.concat(config_dataset.root_path, fname)
    local outpath =
        paths.concat(config_dataset.dest_path, fname
                         .. '_thresh=' .. opt.threshold .. '.th7')
    if not paths.filep(outpath) or isdict == false then
        print('[[ generating training data using new dictionary ]]')
        tokenizer.tokenize(dict, inpath, outpath, config_dataset, false)
    end
end


-- returns the datasource and the corresponding dictionary
function dtools.load_dataset(config)
    if config.name == 'ptb' then
        local rpath = './data/ptb'
        local trfname = 'ptb.train.txt_thresh=0.th7'
        local tefname = 'ptb.test.txt_thresh=0.th7'
        local vlfname = 'ptb.valid.txt_thresh=0.th7'
        local dcfname = 'ptb.dictionary_nclust=100_thresh=0.th7'
        return WordDataset{
            root_path  = rpath,
            train_file = trfname,
            test_file  = tefname,
            valid_file = vlfname,
            dicts = dcfname,
            batch_size = config.batch_size,
            train_nshards = 1
        },
        torch.load(paths.concat(rpath, dcfname))

    elseif config.name == 'text8' then
        local rpath = './data/text8'
        local trfname = 'text8.train.txt_thresh=0.th7'
        local vlfname = 'text8.valid.txt_thresh=0.th7'
        local dcfname = 'text8.dictionary_nclust=210_thresh=0.th7'
        return WordDataset{
            root_path = rpath,
            train_file = trfname,
            valid_file = vlfname,
            dicts = dcfname,
            batch_size = config.batch_size,
            train_nshards = 1
        },
        torch.load(paths.concat(rpath, dcfname))

    else
        error('Datasets: Unknown dataset ' .. config.name)
    end
end

return dtools
