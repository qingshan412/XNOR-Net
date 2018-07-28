--  th main.lua -data ../../../data/ImageNet/xnor256 -nGPU 1 
-- -batchSize 128 -netType alexnet -binaryWeight -dropout 0.1
-- 
--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
-- th main.lua -data [path to ImageNet dataset] -nGPU 1 -batchSize 128 
-- -netType alexnet -binaryWeight -dropout 0.1
--
-- th main.lua -data [path to ImageNet dataset] -nGPU 4 -batchSize 800 
-- -netType alexnetxnor -binaryWeight -optimType adam -epochSize 1500
-- 
-- To use the trained models use the option -retrain [path to the trained model file] 
-- and -testOnly
-- 
-- find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}
--
-- debugger = require('fb.debugger')
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'newLayers'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

print('parser ready...')
os.exit()

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')

opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')


epoch = opt.epochNumber
if opt.testOnly then
	test()
else
  for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
  end
end