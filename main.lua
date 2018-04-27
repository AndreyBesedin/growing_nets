require 'torch'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
dofile('./lib_stream.lua')
torch.setdefaulttensortype('torch.FloatTensor')
opt = {
  batchSize = 60,
  valBatchSize = 60,
  batches_per_env = 1000,
  epochs = 1000,
  lrC = 0.005,
  lrW = 1e+2,
  nb_classes = 4,
  load_pretrained_classifier = true,
  load_pretrained_weighter = false,
  train = 'full',
  cuda = true,
  dataset = 'MNIST', -- MNIST 
  optimizeW = true,
  nb_env = 50
}
opt.manualSeed = torch.random(1, 10000) -- fix seed
torch.manualSeed(opt.manualSeed)

train_full = torch.load('./datasets/cifar10/trainset_by_class.t7')
test_full = torch.load('./datasets/cifar10/testset_by_class.t7')

--------------------------------------------------------------------------------------------------------------------------------
-- DEFINING THE CLASSIFIER
--------------------------------------------------------------------------------------------------------------------------------
function initC(nc, firstConv, secondConv, nbClasses)
  local netC = nn.Sequential()
  netC:add(nn.SpatialConvolution(nc, firstConv, 4, 4, 2, 2)):add(nn.ReLU(true))
  netC:add(nn.SpatialMaxPooling(2, 2, 2, 2))  
  netC:add(nn.SpatialConvolution(firstConv, secondConv, 4, 4, 2, 2)):add(nn.ReLU(true)):add(nn.View(4*secondConv))
  netC:add(nn.Linear(4*secondConv,nbClasses)):add(nn.LogSoftMax())
  return netC
end

function count_elements(Tensor)
  local s = Tensor:size()
  local tot = 1
  for idx = 1, s:size(1) do 
    tot = tot*s[idx] 
  end 
  return tot
end

function getData(full_data, env)
  local data = {}; data.data = {}; data.labels = {}
  local count = 0
  for idx = 1, 10 do
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


--------------------------------------------------------------------------------------------------------------------------------
-- TRAINING on classes 1 and 2
--------------------------------------------------------------------------------------------------------------------------------

-- netC = initC(16, 32, opt.nb_classes)
-- pC, gpC = netC:getParameters()
-- pC = pC:normal(0, 1)
netC_old = torch.load('./pretrained_models/netC12.t7')
netC_old = netC_old:cuda()
pC_old, gpC_old = netC_old:getParameters()
p_old, gp_old  = netC_old:parameters()
opt.nb_classes = 2
--netC = initC(3, 18, 35, opt.nb_classes)
netC = initC(3, 16, 32, opt.nb_classes)
if opt.cuda then netC = netC:cuda() end
pC, gpC = netC:getParameters()
--pC:normal(0,1)
pC:normal(0,1)
p, gp  = netC:parameters()
--p[1][{{1,16},{},{},{}}]     = p_old[1]
--p[2][{{1,16}}]              = p_old[2]
--p[3][{{1,32},{1,16},{},{}}] = p_old[3]
--p[4][{{1,32}}]              = p_old[4]
--p[5][{{1,2},{1,128}}]       = p_old[5]
--p[6][{{1,2}}]               = p_old[6]

optimState = {
  learningRate = 0.0001,
  learningWeightDecay = 0
}
function get_vect_classes(classes)
  local res = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  for idx = 1, #classes do res[classes[idx]] = 1 end
  return res
end

--confusion_train = optim.ConfusionMatrix(opt.nb_classes)
critC = nn.ClassNLLCriterion(); 
critC = critC:cuda()
current_classes = get_vect_classes({1, 2})
trainset = getData(train_full, current_classes)
testset = getData(test_full, {1, 1, 0, 0, 0, 0, 0, 0, 0, 0})
confusion = test_classifier(netC, testset, opt); print(confusion)
for epoch = 1, opt.epochs do
  local indices = torch.randperm(trainset.data:size(1))
  for idx = 1, trainset.data:size(1), opt.batchSize do
    xlua.progress(idx, trainset.data:size(1))
    local feval = function(x)
      if x ~= pC then pC:copy(x) end
      gpC:zero()
      local outputs = netC:forward(batch.data)
      local f = critC:forward(outputs, batch.labels)
      local df_do = critC:backward(outputs, batch.labels)
      netC:backward(batch.data, df_do)
--      gp[1][{{1,16},{},{},{}}]:zero()
--      gp[2][{{1,16}}]:zero()
--      gp[3][{{1,32},{},{},{}}]:zero()
--      gp[4][{{1,32}}]:zero()
--      gp[5][{{1,2},{}}]:zero()
--      gp[6][{{1,2}}]:zero()
      --confusion_train:batchAdd(outputs, batch.labels)
      return f,gpC
    end    
    batch = getBatch(trainset, indices[{{idx, math.min(idx + opt.batchSize - 1, trainset.data:size(1))}}]:long())
    optim.adam(feval, pC, optimState)
    --optim.sgd(feval, pC, optimState)
--    if (idx-1)%(100*opt.batchSize)==0 then confusion = test_classifier(netC, testset, opt); print(confusion) end
  end
  --confusion_train:zero()
  --confusion_train:updateValids()
--  print('Training confusion: '); print(confusion_train)
  print('Epoch ' .. epoch .. ' is over, testing')
  confusion = test_classifier(netC, testset, opt); print(confusion)
end

