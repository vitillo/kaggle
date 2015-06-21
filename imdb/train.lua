require 'nn'
require 'image'
require 'cutorch'
require 'cunn'
require 'optim'
require 'csvigo'

torch.setdefaulttensortype('torch.FloatTensor')

function load_data(path)
  local raw_data = csvigo.load{path=path, mode="raw"}
  local data = torch.Tensor(#raw_data, #raw_data[1])

  for i = 1, #raw_data do
    data[i] = torch.Tensor(raw_data[i]) + 1  -- Python indexing starts from zero...
  end

  local labels = data[{{}, -1}]
  local predictors = data[{{}, {1, -2}}]

  return {labels = labels,
          predictors = predictors,
          size = function() return (#labels)[1] end}
end

function setup_model(vocab_size, window_size, num_hidden_1, num_hidden_2)
  local model = nn.Sequential()
  local criterion = nn.ClassNLLCriterion()

  model:add(nn.LookupTable(vocab_size, num_hidden_1))
  model:add(nn.View(window_size*num_hidden_1))
  model:add(nn.Linear(window_size*num_hidden_1, num_hidden_2))
  model:add(nn.Sigmoid())
  model:add(nn.Linear(num_hidden_2, vocab_size))
  model:add(nn.LogSoftMax())

  return model, criterion
end

function minibatch(train_data, shuffle, t, batchSize)
    local inputs = {}
    local targets = {}

    for i = t,math.min(t + batchSize - 1, train_data:size()) do
        local input = train_data.predictors[shuffle[i]]
        local target = train_data.labels[shuffle[i]]

        table.insert(inputs, input)
        table.insert(targets, target)
    end

    return inputs, targets
end


function train_epoch(train, classes)
    epoch = epoch or 1

    local time = sys.clock()
    local shuffle = torch.randperm(train:size())
    local parameters, gradParameters = model:getParameters()
    local confusion = optim.ConfusionMatrix(classes)

    local batchSize = 25
    local optimState = {
        learningRate = 1e-3,
        weightDecay = 0.001,
        momentum = 0.9,
        learningRateDecay = 1e-7
    }

    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

    for t = 1,train:size(),batchSize do
        inputs, targets = minibatch(train, shuffle, t, batchSize)

        local feval = function(x)
            -- get new parameters
            if x~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- evaluate function for complete mini batch
            for i = 1,#inputs do
               -- estimate f
                local output = model:forward(inputs[i])
                local err = criterion:forward(output, targets[i])
                f = f + err

                -- estimate df/dW
                local df_do = criterion:backward(output, targets[i])
                model:backward(inputs[i], df_do)

                -- update confusion
                confusion:add(output, targets[i])
            end

            -- normalize gradients and f(X)
            gradParameters:div(#inputs)
            f = f/#inputs

            -- return f and df/dW
            return f,gradParameters
        end

        optim.sgd(feval, parameters, optimState)
    end

    -- time taken
    time = sys.clock() - time
    time = time / train:size()
    print("==> time to learn 1 sample = " .. (time*1000) .. ' ms')

    confusion:updateValids()
    print("==> average row correct: " .. 100*confusion.averageValid)
    confusion:zero()

    epoch = epoch + 1
end

function map(func, array)
  local new_array = {}
  for i,v in ipairs(array) do
    new_array[i] = func(v)
  end
  return new_array
end

n_classes = 1000
window_size = 4

classes = map(tostring, torch.totable(torch.range(1, n_classes)))
train_data = load_data("data.csv")
model, criterion = setup_model(n_classes, window_size, 30, 500)

for i=1,10 do
  train_epoch(train_data, classes)
end

torch.save("model", model)
