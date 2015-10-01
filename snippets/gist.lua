require 'cudnn'
require 'fbcunn'

local timer = torch.Timer()

local tensorsAreProbablySimilar = function(l, r, epsilon)
    epsilon = epsilon or 0.00001
    return math.abs((l:norm() - r:norm()) / (l:norm() + r:norm())) < epsilon
end

-- Create sample net (same as overfeat l2)
local net = nn.Sequential()
net:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1))
net:add(cudnn.ReLU(true))
net:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
net:cuda()

-- Build 2-GPU version
local dp = nn.DataParallel(1)
dp:add(net:clone())
-- dp:add(net:clone())
dp:cuda()

-- Batch input
local input = torch.CudaTensor(100, 96, 26, 26)
cutorch.synchronize()

-- Test 1-GPU (do 20 times)
timer:reset()
local res
for i=1,20 do
  res = net:forward(input)
end
cutorch.synchronize()
print(string.format("1-GPU in %0.3fs", timer:time().real))

-- Test 2-GPU (do 20 times)
timer:reset()
local res2
for i=1,20 do
  res2 = dp:forward(input)
end
cutorch.synchronize()
print(string.format("2-GPU in %0.3fs", timer:time().real))

-- Compare 2 versions
assert(tensorsAreProbablySimilar(res, res2))
