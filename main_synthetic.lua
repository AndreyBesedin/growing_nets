require 'torch'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'gnuplot'
dofile('./lib_stream.lua')
dofile('./generate_synthetic.lua')
torch.setdefaulttensortype('torch.FloatTensor')
opt = {
  batchSize = 100,
  valBatchSize = 60,
  batches_per_env = 1000,
  epochs = 50,
  nb_classes = 4,
  load_pretrained_classifier = true,
  train = 'full',
  cuda = true,
  nb_env = 50
}
opt.manualSeed = torch.random(1, 10000) -- fix seed
torch.manualSeed(opt.manualSeed)

gen_params = {
  {mu = torch.FloatTensor({0, 0}),  sigma = torch.eye(2)},
  {mu = torch.FloatTensor({4, 3}),  sigma = torch.eye(2)*2},
  {mu = torch.FloatTensor({7,-6}),  sigma = torch.eye(2)*3},
  {mu = torch.FloatTensor({-3,-7}), sigma = torch.eye(2)*2},
  {mu = torch.FloatTensor({-6, 4}), sigma = torch.eye(2)*1},
}

train_full = generate_labeled_dataset(10000, gen_params)
visu_set = generate_2D_testset(10000, -10,-10, 10, 10)
--------------------------------------------------------------------------------------------------------------------------------
-- DEFINING THE CLASSIFIER
--------------------------------------------------------------------------------------------------------------------------------
function initC(layers)
  -- init layers = [32, 128, 64]
  local netC = nn.Sequential()
  netC:add(nn.Linear(2, layers[1])):add(nn.ReLU(true))
  netC:add(nn.Linear(layers[1], layers[2])):add(nn.ReLU(true))
  netC:add(nn.Linear(layers[2], layers[3])):add(nn.ReLU(true))
  netC:add(nn.Linear(layers[3], layers[4])):add(nn.LogSoftMax())
  return netC
end

function getData(full_data, env)
  local data = {}; data.data = {}; data.labels = {}
  local count = 0
  for idx = 1, #full_data do
    if env[idx] > 0 then
      count = count + 1
      data.data[count] = full_data[idx]
      data.labels[count] = torch.zeros(data.data[count]:size(1)):fill(idx)
    end
  end
  data.data = torch.cat(data.data,1)
  data.labels = torch.cat(data.labels,1)
  return data
end

function visualize_testset_with_labels(model, testset)
  local res_visu = model:forward(testset:cuda())
  _, labels = torch.max(res_visu, 2)
  labels = labels:squeeze()
  nb_classes = labels:max()
  labels = labels:long()
  to_plot = {}
  colors = {'red', 'blue', 'green', 'yellow', 'black'}
  for idx = 1, nb_classes do
    ids = torch.range(1,labels:nElement())[labels:eq(idx)]
    if #ids:size()>0 then
      set = testset:index(1, ids:long()):transpose(1,2)
      to_plot[idx] = {set[1], set[2], 'with points lc rgb "' .. colors[idx] .. '"'}
    else 
      to_plot[idx] = {torch.FloatTensor({10}), torch.FloatTensor({10}), 'with points lc rgb "' .. colors[idx] .. '"'}
    end
  end
  return to_plot
end  
  
function clone_params(params)
  local res = {}
  for idx = 1, #params do
    res[idx] = params[idx]:clone()
  end
  return res
end
--------------------------------------------------------------------------------------------------------------------------------
-- TRAINING on classes 1 and 2
--------------------------------------------------------------------------------------------------------------------------------

optimState = {
  learningRate = 0.0001,
  learningWeightDecay = 0
}

--confusion_train = optim.ConfusionMatrix(opt.nb_classes)
critC = nn.ClassNLLCriterion(); 
critC = critC:cuda()
train_full = generate_labeled_dataset(10000, gen_params)
train_scenario = {
    {1, 1, 0, 0, 0},
    {0, 1, 1, 0, 0},
    {0, 0, 1, 1, 0},
    {0, 0, 0, 1, 1},
    {1, 0, 0, 0, 1}
}

old_layers = {32, 128, 64, 2}
new_layers = {32, 128, 64, 2}

netC = initC(new_layers)
pC, gpC = netC:getParameters()
pC:normal(0,1)
p, gp  = netC:parameters()
if opt.cuda then netC = netC:cuda() end
for scenario_nb = 1, #train_scenario do
  optimState = {
    learningRate = 0.0001,
    learningWeightDecay = 0
  }
  trainset = getData(train_full, train_scenario[scenario_nb])
  if scenario_nb > 1 then
    netC = initC(new_layers)
    pC, gpC = netC:getParameters()
    p, gp  = netC:parameters()
    pC:normal(0,1)
    if opt.cuda then netC = netC:cuda() end
    p[1][{{1,old_layers[1]},{}}]                 = p_old[1]
    p[2][{{1,old_layers[1]}}]                    = p_old[2]
    p[3][{{1,old_layers[2]},{1, old_layers[1]}}] = p_old[3]
    p[4][{{1,old_layers[2]}}]                    = p_old[4]
    p[5][{{1,old_layers[3]},{1, old_layers[2]}}] = p_old[5]
    p[6][{{1,old_layers[3]}}]                    = p_old[6]
    p[7][{{1,old_layers[4]},{1, old_layers[3]}}] = p_old[7]
    p[8][{{1,old_layers[4]}}]                    = p_old[8]
  end
  
  for epoch = 1, opt.epochs do
    local indices = torch.randperm(trainset.data:size(1))
    for idx = 1, trainset.data:size(1), opt.batchSize do
      pC, gpC = netC:getParameters()
      xlua.progress(idx, trainset.data:size(1))
      local feval = function(x)
        if x ~= pC then pC:copy(x) end
        gpC:zero()
        local outputs = netC:forward(batch.data)
        local f = critC:forward(outputs, batch.labels)
        local df_do = critC:backward(outputs, batch.labels)
        netC:backward(batch.data, df_do)
        if scenario_nb > 1 then
          gp[1][{{1,old_layers[1]},{}}]:zero()
          gp[2][{{1,old_layers[1]}}]:zero()
          gp[3][{{1,old_layers[2]},{}}]:zero()
          gp[4][{{1,old_layers[2]}}]:zero()
          gp[5][{{1,old_layers[3]},{}}]:zero()
          gp[6][{{1,old_layers[3]}}]:zero()
          gp[7][{{1,old_layers[4]},{}}]:zero()
          gp[8][{{1,old_layers[4]}}]:zero()
        end
        --confusion_train:batchAdd(outputs, batch.labels)
        return f,gpC
      end    
      batch = getBatch(trainset, indices[{{idx, math.min(idx + opt.batchSize - 1, trainset.data:size(1))}}]:long())
      optim.adam(feval, pC, optimState)
      --print(gpC:norm())
      --optim.sgd(feval, pC, optimState)
      if (idx-1)%(50*opt.batchSize)==0 then   
          res_visu = visualize_testset_with_labels(netC, visu_set)
          gnuplot.plot(res_visu) 
      end
    end
    --confusion_train:zero()
    --confusion_train:updateValids()
--  print('Training confusion: '); print(confusion_train)
    print('Epoch ' .. epoch .. ' is over')
-- confusion = test_classifier(netC, testset, opt); print(confusion)
  end
  old_layers = {new_layers[1], new_layers[2], new_layers[3], new_layers[4]}
  if scenario_nb < 5 then
    new_layers = {old_layers[1]+5, old_layers[2]+10, old_layers[3]+10, math.min(5, scenario_nb + 2)}
  end
  p_old = clone_params(p)
  netC:clearState()
end

