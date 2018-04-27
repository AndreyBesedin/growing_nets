require 'distributions'
torch.setdefaulttensortype('torch.FloatTensor')

function generate_2D_testset(data_size, x1,y1, x2, y2)
  local res = torch.FloatTensor(2, data_size)
  res[1]:uniform(x1, x2); res[2]:uniform(y1, y2)
  return res:transpose(1,2)
end

function generate_class(nbSamples, p)
  local res = torch.FloatTensor(nbSamples, 2)
  for idx = 1, nbSamples do
    res[idx] = distributions.mvn.rnd(p.mu, p.sigma)
  end
  return res
end

function generate_labeled_dataset(samples_per_class, params)
  local res = {}
  for idx = 1, #params do
    res[idx] = generate_class(samples_per_class, params[idx])
  end
  return res
end

--testset = generate_2D_testset(10000, -10,-10, 10, 10)

-- Trainset parameters
-- params = {
--   {mu = torch.FloatTensor({0, 0}),  sigma = torch.eye(2)},
--   {mu = torch.FloatTensor({4, 3}),  sigma = torch.eye(2)*2},
--   {mu = torch.FloatTensor({7,-6}),  sigma = torch.eye(2)*3},
--   {mu = torch.FloatTensor({-3,-7}), sigma = torch.eye(2)*2},
--   {mu = torch.FloatTensor({-6, 4}), sigma = torch.eye(2)*1},
-- }

-- dataset = generate_labeled_dataset(10000, params)
-- 
-- gnuplot.plot(
--   {dataset[1][1],dataset[1][2], 'with points lc rgb "red"'},
--   {dataset[2][1],dataset[2][2], 'with points lc rgb "blue"'},
--   {dataset[3][1],dataset[3][2], 'with points lc rgb "green"'},
--   {dataset[4][1],dataset[4][2], 'with points lc rgb "yellow"'},
--   {dataset[5][1],dataset[5][2], 'with points lc rgb "black"'}
-- )
-- sample = distributions.mvn.rnd(mu, sigma)

