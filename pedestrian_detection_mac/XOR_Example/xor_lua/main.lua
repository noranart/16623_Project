require 'torch'
require 'nn'

model = ""

function loadNeuralNetwork(path)
    --print (path)
    --print ("Loaded Neural Network -- Success")
    model = torch.load(path)
end

function classifyExample(tensorInput)
    --[[
    print(tensorInput[1])
    print(tensorInput[2])
    print(tensorInput[3])
    print(tensorInput[4])
    print(tensorInput[5])
    ]]--

    x = torch.FloatTensor(1,113,113,3)
    x:copy(tensorInput)
    x = x:permute(1,4,2,3):float()/255.0

    --torch.save('/Users/saintsol/Desktop/1.t7',tensor)

    mean = torch.Tensor{0.485, 0.456, 0.406}
    std = torch.Tensor{0.229, 0.224, 0.225}
    for i=1,3 do -- channels
        x:narrow(i,1,1):add(-mean[i])
        x:narrow(i,1,1):div(std[i])
    end


    v = model(x)
    return v[2]
end