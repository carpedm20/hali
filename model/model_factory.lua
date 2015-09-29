local models = {}

function models.makeModel(params, dict, n_classes)
  local n_classes = n_classes or dict.index_to_freq:size(1)
  local n_hidden = params.n_hidden
  local enc, dec, dec_loss
  local internal_layers = {}

  if string.find(params.name, 'lstm') then
    -- input : a table {x_t, {h_{t-1}, c_{t-1}}}
    -- x_t : current input (1-of-N vector)
    -- h_{t-1} : previous set of hidden units
    -- c_{t-1} : memory units
    -- 
    -- Detailed computation :
    -- i = logistic(W_{xi} x_t + W_{hi} h_{t-1})
    -- f = logistic(W_{xf} x_t + W_{hf} h_{t-1})
    -- o = logistic(W_{xo} x_t + W_{ho} h_{t-1})
    -- g = th(W_{xg} x_t + W_{hg} H_{t-1})
    -- c_t = f .* c_{t-1} + i .* g
    -- h_t = o .* th(c_t)

    -- output : a table {h_t, c_t}
    -- h_t : current hidden state
    -- c_t : updated memory units

    -- In a batch setting,
    -- B : mini-batch size
    -- D : size of hidden/memory state
    -- x_t : D dimension
    -- h_{t-1}, c_{t-1} : BxD dimension

    -- from fbcunn
    -- Fast lookup table, supporting both CPU and GPU modes.
    -- http://facebook.github.io/fbcunn/fbcunn/#fbcunn.fbcunn.LookupTableGPU.dok
    local embed1 = nn.LookupTableGPU(n_classes, n_hidden)
    local embed2 = nn.LookupTableGPU(n_classes, n_hidden)
    local embed3 = nn.LookupTableGPU(n_classes, n_hidden)
    local embed4 = nn.LookupTableGPU(n_classes, n_hidden)

    local project1 = nn.LookupTableGPU(n_hidden, n_hidden)
    local project2 = nn.LookupTableGPU(n_hidden, n_hidden)
    local project3 = nn.LookupTableGPU(n_hidden, n_hidden)
    local project4 = nn.LookupTableGPU(n_hidden, n_hidden)

    -- Input Encoder: {x_t, {h_{t-1}, c_{t-1}}}
    -- Output Encoder: {h_t, c_t}
    -- Input Decoder: h_t
    -- Output Decoder: o_t

    -- construct LSTM graph: encoder
    local lstm_symbol = nn.Identity()()
    local lstm_prev_state = nn.Identity()()

    -- split into two tables
    local prev_hidden, prev_memory = lstm_prev_state:split(2)
    local embed1n = embed1(lstm_symbol)
    local embed2n = embed2(lstm_symbol)
    local embed3n = embed3(lstm_symbol)
    local embed4n = embed4(lstm_symbol)
