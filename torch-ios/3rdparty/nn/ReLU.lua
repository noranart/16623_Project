local ReLU, parent = torch.class('nn.ReLU','nn.Module')

function ReLU:__init()
   parent.__init(self)
   self.threshold = 0 --or 1e-6
   self.val = 0
   
   self:reset()

end

function ReLU:updateOutput(input)
   
   	print('#### RELU in ####')
	print(#input)
	print('#### end RELU in ####')
	
   input.nn.Threshold_updateOutput(self, input)
   
    print('#### RELU out ####')
	--print(#self.output)
	print(self.output[1][1][1][1])
	print(self.output[1][1][1][2])
	print(self.output[1][1][1][3])
	print(self.output[1][1][1][4])
	print(self.output[1][1][1][5])

	print('#### end RELU out ####')

   
   return self.output
end

function ReLU:updateGradInput(input, gradOutput)
   input.nn.Threshold_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
