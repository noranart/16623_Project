--[[
local BN, parent = torch.class('nn.SpatialBatchNormalization', 'nn.Module')

function SpatialBatchNormalization:__init()
    Parent.__init(self,0,0)
end

function SpatialBatchNormalization:updateOutput(input)
   
   	print('#### batch in ####')
	print(#input)
	print('#### end batch in ####')
	
   input.nn.SpatialBatchNormalization_updateOutput(self, input)
  
  	print('#### batch out ####')
	print(#self.output)
	print('#### end batch out ####')

   
   return self.output
end
]]--

local BN,parent = torch.class('nn.SpatialBatchNormalization', 'nn.Module')

function BN:__init(nOutput, eps, momentum, affine)
   parent.__init(self)
   assert(nOutput and type(nOutput) == 'number',
          'Missing argument #1: dimensionality of input. ')
   assert(nOutput ~= 0, 'To set affine=false call BatchNormalization'
     .. '(nOutput,  eps, momentum, false) ')
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end
   self.eps = eps or 1e-5
   self.momentum = momentum or 0.1
   self.running_mean = torch.zeros(nOutput)
   self.running_var = torch.ones(nOutput)

   if self.affine then
      self.weight = torch.Tensor(nOutput)
      self.bias = torch.Tensor(nOutput)
      self.gradWeight = torch.Tensor(nOutput)
      self.gradBias = torch.Tensor(nOutput)
      self:reset()
   end
end

function BN:updateOutput(input)
   input.nn.SpatialBatchNormalization_updateOutput(self, input)

   
   return self.output
end

