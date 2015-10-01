require 'cudnn'
require 'fbcunn'

timer = torch.Timer()

net = nn.Sequential()
n_classes, n_hidden = 10, 20

net_parallel = nn.ParallelTable()
-- https://github.com/torch/nn/blob/master/doc/convolution.md#lookuptable
embed = nn.LookupTableGPU(n_classes, n_hidden)
project = nn.LinearNB(n_hidden, n_hidden)

net_parallel:add(embed)
net_parallel:add(project)

net:add(net_parallel)
net:add(nn.CAddTable())
net:add(nn.Threshold())
net:cuda()

dp = nn.DataParallel(1)
dp:add(net:clone())
dp:cuda()

input1 = torch.CudaTensor{1,2,1,1,1}
input2 = torch.CudaTensor(5, n_hidden)
cutorch.synchronize()

timer:reset()
for i=1, 20 do
  res = net:forward{input1, input2}
end
cutorch.synchronize()
print(string.format("GPU1 in %0.3fs", timer:time().real))

timer:reset()
local res2
for i=1, 20 do
  res2 = dp:forward{input1, input2}
end
cutorch.synchronize()
print(string.format("GPU2 in %0.3fs", timer:time().real))

tensorsAreProbablySimilar = function(l, r, epsilon)
  epsilon = epsilon or 0.00001
  return math.abs((l:norm() - r:norm()) / (l:norm() + r:norm())) < epsilon
end

assert(tensorsAreProbablySimilar(res, res2))
