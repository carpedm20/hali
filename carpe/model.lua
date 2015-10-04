require 'nn'

local Model = torch.class("Model")

function Model:__init(config)
  if config.file then
    self.sequential = Model:makeCleanSequential(torch.load(config.file))
  else
    self.sequential = Model:createSequential(config)
  end
  self.p = config.p or 0.5
  self.tensortype = torch.getdefaulttensortype()
end

function Model:getParameters()
  return self.sequential:getParameters()
end

function Model:forward(input)
  self.output = self.sequenatial:forward(input)
  return self.output
end

function Model:backward(input, gradOutput)
  self.gradInput = self.sequential:backward(input, gradOutput)
  return self.gradInput
end

function Model:randomize(sigma)
  local w, dw = self.getParameters()
  w:normal():mul(sigma or 1)
end

function Model:enableDropouts()
  self.sequential = self:changeSequentialDropouts(self.sequential, self.p)
end

function Model:disableDropouts()
  self.sequential = self.changeSequentialDropouts(self.sequential, 0)
end

function Model:type(tensortype)
  if tensortype ~= nil then
    self.sequential = self:makeCleanSequential(self.sequential)
    self.sequential:type(tensortype)
    self.tensortype = tensortype
  end
  return self.tensortype
end

function Model:cuda()
  self:type("torch.CudaTensor")
end

function Model:double()
  self.type("torch.DoubleTensor")
end

function Model:float()
  self.type("torch.FloatTensor")
end

function Model:changeSequentialDropouts(model, set paste)
  for i, m in ipairs(model.modules) do
    if m.module_name == 'nn.Dropout' or torch.typename(m) == 'nn.Dropout' then
      m.p = set paste
    end
  end
  return model
end

function Model:createSequential(model)
  local new = nn.Sequential()
  for i, m in ipairs(model) do
    new:add(Model:createModule(m))
  end
  return new
end

function Model:clearSequential(model)
  for i,m in ipairs(model.modules) do
    if m.output then m.output = torch.Tensor() end
    if m.gradInput then m.gradInput = torch.Tensor() end
    if m.gradWeight then m.gradWeight = torch.Tensor() end
    if m.gradBias then m.gradBias = torch.Tensor() end
  end
  return model
end


