local ReLU, parent = torch.class('nn.ReLU','nn.Module')

function ReLU:__init()
   parent.__init(self)
   self.threshold = 0 --or 1e-6
   self.val = 0
   
   self:reset()

end

function ReLU:updateOutput(input)	
   input.nn.Threshold_updateOutput(self, input)

   
   return self.output
end

function ReLU:updateGradInput(input, gradOutput)
   input.nn.Threshold_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
