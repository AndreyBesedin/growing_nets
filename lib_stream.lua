-- usefull functions for stream training
function generate_2D_quaters(data_size, env, opt)
  --env = {e1,e2,e3,e4}, ei = 1 if class is present and 0 otherwise
  local data = {}
  local eps = 1e-15
  box_size = 100
  env = torch.totable(env)
  labels = {}; idx_lab = 1; for idx = 1, #env do if env[idx]==1 then labels[idx_lab] = idx; idx_lab = idx_lab+1 end; end
  class_size = math.floor(data_size[1]/#labels)
  data.data = torch.zeros(data_size[1], data_size[2])
  data.labels = torch.zeros(data_size[1])
  for idx_class = 1, #labels do
    local l = labels[idx_class]
    box = (l==1 and {0, box_size, eps, box_size}) or 
          (l==2 and {eps, box_size, -box_size, 0}) or
          (l==3 and {-box_size, 0, -box_size, -eps}) or
          (l==4 and {-box_size, 0, 0, box_size})
    data.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{1}}] = data.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{1}}]:uniform(box[1], box[2]); 
    data.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{2}}] = data.data[{{1 + (idx_class-1)*class_size, idx_class*class_size},{2}}]:uniform(box[3], box[4]); 
    data.labels[{{1 + (idx_class-1)*class_size, idx_class*class_size}}]:fill(labels[idx_class])
  end
  if opt.cuda then data.data = data.data:cuda(); data.labels = data.labels:cuda() end
  return data
end

function getEnv(opt)
  local data = torch.rand(opt.nb_classes)
  _,idx_max = torch.max(data, 1); data[idx_max[1]] = 1
  local res = data:ge(0.5):float()
  if opt.cuda then res = res:cuda() end
  return res
end

function getBatch(data, indices_)
  local batch = {}
  batch.data = data.data:index(1, indices_:long())
  batch.labels = data.labels:index(1, indices_:long())
  if opt.cuda then batch.data = batch.data:cuda(); batch.labels = batch.labels:cuda() end
  return batch
end

function test_classifier(C, data, opt)
  local confusion = optim.ConfusionMatrix(opt.nb_classes)
  confusion:zero()
  for idx = 1, data.data:size(1), opt.batchSize do
    --xlua.progress(idx, opt.testSize)
    indices_ = torch.range(idx, math.min(idx + opt.batchSize, data.data:size(1)))
    local batch = getBatch(data, indices_:long())
    if opt.cuda then batch.data = batch.data:cuda() else batch.data = batch.data:float(); C = C:float() end
    local y = C:forward(batch.data)
    y = y:float()
    _, y_max = y:max(2)
    confusion:batchAdd(y_max:squeeze():float(), batch.labels:float())
  end
  confusion:updateValids()
  return confusion  
end

