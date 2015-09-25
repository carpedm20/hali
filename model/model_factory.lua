local models = {}

function models.makeModel(params, dict, n_classes)
  local n_classes = n_classes or dict.index_to_freq:size(1)
