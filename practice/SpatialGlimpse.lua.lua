-- code referenced from https://github.com/clementfarabet/lua---nnx/blob/master/SpatialReSampling.lua
-- code referenced from https://github.com/nicholas-leonard/dpnn/blob/master/SpatialGlimpse.lua
local SpatialGlimpse.lua, parent = torch.class("nn.SpatialGlimpse.lua", "nn.Module")

function SpatialReSampling:__init(...)
  parent.__init(self)
end

function SpatialReSampling:updateOutput(input)
  -- input : 3d or 4d tensor ([batchSize x nInputPlane x width x height])
  -- The re-sampling is done using bilinear interpolation.
  local hDim, wDim = 2,3
  if input:dim() == 4 then
    hdim, wDim = 3, 4
  end
  self.oheight = self.oheight or self.rheight*input:size(hDim)
  self.owidth = self.owidth or self.rwidth*input:size(wDim)
  input.nn.Spatial

function SpatialGlimpse.lua:__init(size, depth, scale)
  self.size = size
  self.depth = depth or 3
  self.scale = scale or 2

  assert(torch.type(self.size) == 'number')
  assert(torch.type(self.depth) == 'number')
  assert(torch.type(self.scale) == 'number')
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  self.module = 
