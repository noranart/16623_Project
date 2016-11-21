require 'torch'
require 'nn'

model = ""

function loadNeuralNetwork(path)
    print (path)
    print ("Loaded Neural Network -- Success")
    model = torch.load(path)
end

function classifyExample(tensorInput)
    tensorInput = torch.reshape(tensorInput, torch.LongStorage{1, 113, 113, 3})
    tensor = torch.FloatTensor(tensorInput):float():permute(1, 4, 2, 3)/255.0


    mean = torch.Tensor{0.485, 0.456, 0.406}
    std = torch.Tensor{0.229, 0.224, 0.225}
    for i=1,3 do -- channels
        tensor:narrow(i,1,1):add(-mean[i])
        tensor:narrow(i,1,1):div(std[i])
    end


    v = model(tensor)
    print("predict: ", v[2])
    return v[2]
end