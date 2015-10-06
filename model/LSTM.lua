local LSTM = torch.class('LSTM')

function LSTM.lstm(config)
  local n_hidden = config.n_hidden
  local n_classes = config.n_classes
  local n_layers = config.n_layers
  local gpu_mode = config.gpu_mode or false

  net = nn.Sequential()

  if gpu_mode then
    LookupTable = nn.LookupTable
  else
    LookupTable = nn.LookupTableGPU
  end

  for i=1, n_layers do
    -- H (n_hidden) : hidden unit size
    -- D (n_classes) : word vector size
    -- B (b_size) : batch size
    local input = nn.Identity()() -- [D * B]
    local lstm_prev_state = nn.Identity()() -- [2H * B]

    local prev_h, prev_c = lstm_prev_state:split(2)

    -- one-hot-vector       x weight                 = output
    -- [b_size * n_classes] x [n_classes * n_hidden] = [b_size * n_hidden]
    local embed1 = LookupTable(n_classes, n_hidden)
    local embed2 = LookupTable(n_classes, n_hidden)
    local embed3 = LookupTable(n_classes, n_hidden)
    local embed4 = LookupTable(n_classes, n_hidden)

    -- [b_size * n_hidden] x [n_hidden * n_hidden] = [b_size * n_hidden]
    local hidden1 = nn.LinearNB(n_hidden, n_hidden)
    local hidden2 = nn.LinearNB(n_hidden, n_hidden)
    local hidden3 = nn.LinearNB(n_hidden, n_hidden)
    local hidden4 = nn.LinearNB(n_hidden, n_hidden)

    -- [H * D] x [D * B] = [H * B]
    local output1 = embed1(input)
    local output2 = embed2(input)
    local output3 = embed3(input)
    local output4 = embed4(input)

    -- [H * H] x [H * B] = [H * B]
    local h_output1 = hidden1(prev_h)
    local h_output2 = hidden2(prev_h)
    local h_output3 = hidden3(prev_h)
    local h_output4 = hidden4(prev_h)

   acocal i_gate = nn.Sigmoid()(nn.CAddTable()({output1, h_output1}))
    local f_gate = nn.Sigmoid()(nn.CAddTable()({output2, h_output2}))
    local o_gate = nn.Sigmoid()(nn.CAddTable()({output3, h_output3}))
    local g_gate = nn.Tanh()(nn.CAddTable()({output4, h_output4}))

    next_c = nn.CAddTable()({nn.CMulTable()({f_gate, prev_c}), nn.CMulTable()({i_gate, g_gate})})
    next_h = nn.CMulTable()({o_gate, nn.Tanh()(next_c)})

    local next_state = nn.Identity(){next_h, next_c}
    encoder = nn.gModule({input, lstm_prev_state}, {next_state})

    return encoder
  end
end

return LSTM
