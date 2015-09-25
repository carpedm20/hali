require 'cunn'
require 'ccn2'

-- design information from http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
-- 3x3 conv. kernels – very small
---- less parameters to learn
---- 
-- conv. stride 1 – no loss of information
-- 5 max-pool layers (x2 reduction)
-- 3 fully-connected (FC) layers

-- tutorial : http://cs231n.github.io/convolutional-networks/

function very_deep_model()
  local model = nn.Sequential()
  local final_mlpconv_layer = nil

  ---------------------
  -- convolution layers
  ---------------------

  -- transpose dimensions:
  -- n = nn.Transpose({1,4},{1,3})
  -- will transpose dims 1 and 4, then 1 and 3...
  model:add(nn.Transpose({1,4},{1,3},{1,2}))

  -- ccn2.SpatialConvolution(nInputPlane, nOutputPlane, kH, [dH = 1], [padding = 0], [groups = 1], [partialSum = oH * oH])
  -- kH : kernel height of convolution
  -- dH : the step of the convolution in the height dimension
  -- padding : additional zeros added per height and weight to input planes. (kH-1)/2 is a good number
  model:add(ccn2.SpatialConvolution(3, 64, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(64, 64, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2))
  model:add(nn.Dropout(0.25))

  model:add(ccn2.SpatialConvolution(64,128,3,1,1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(128,128,3,1,1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2))
  model:add(nn.Dropout(0.25))

  model:add(ccn2.SpatialConvolution(128,256,3,1,1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(256,256,3,1,1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(256,256,3,1,1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(256,256,3,1,1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2))
  model:add(nn.Dropout(0.25))

  --------------------------
  -- Fully Connected Layers
  --------------------------

  model:add(ccn2.SpatialConvolution(256,1024,3,1,0))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(ccn2.SpatialConvolution(1024,1024,1,1,0))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))

  model:add(nn.Transpose({4,1},{4,2},{4,3}))

  -- function SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1))
  model:add(nn.Reshape(10))
  model:add(nn.SoftMax())

  return model
end
