require 'torch'

local SpatialConvolution, parent = torch.class('nn.SpatialConvolution', 'nn.Module')

function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.padW = padW or 0
   self.padH = padH or self.padW


   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)

   self:reset()
end

function SpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end) 
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function SpatialConvolution:updateOutput(input)
   
   	print('#### conv in ####')
	print(#input)
	print('#### end conv in ####')

	--[[	
	print('#### conv var ####')
	print(self.padW)
	print(self.padH)
	print(self.kW)
	print(self.kW)
	print(self.dW)
	print(self.dH)
	print('#### end conv var ####')
	--]]
	
	padInput = torch.zeros(input:size(1), input:size(2), input:size(3)+self.padH*2, input:size(4)+self.padW*2)
	padInput:narrow(3, self.padH+1, padInput:size(3)-self.padH*2):narrow(4, self.padW+1, padInput:size(4)-self.padW*2):copy(input)

	
   input.nn.SpatialConvolution_updateOutput(self, padInput)
  
  	print('#### conv out ####')
	--print(#self.output)
	print(self.output[1][1][1][1])
	print(self.output[1][1][1][2])
	print(self.output[1][1][1][3])
	print(self.output[1][1][1][4])
	print(self.output[1][1][1][5])
	--print(self.output:narrow(1,1,1))
	print('#### end conv out ####')

   
   return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolution_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolution_accGradParameters(self, input, gradOutput, scale)
end
