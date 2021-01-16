#Libraries
import torch
import numpy as np

#Get model Function
def getModel(modelName):
    import torchvision as tv 
    loss_func =  torch.nn.CrossEntropyLoss()
    model = None
    
    if modelName == 'alexnet':
        model = tv.models.alexnet(pretrained=True)
    elif modelName == 'squeezenet':
        model = tv.models.squeezenet1_1(pretrained=True)
    elif modelName == 'resnet34':
        model = tv.models.resnet34(pretrained=True)
    elif modelName == 'vgg16':
        model = tv.models.vgg16_bn(pretrained=True)
    elif modelName == 'mobilenet2':
        model = tv.models.mobilenet_v2(pretrained=True)
        
    assert(model != None)
    return model, loss_func

#------------------------------------------------------------------------------------
#Get Featue Maps for compression
def getFMs(model, loss_func, training=True, batchSize=10, safetyFactor=0.75):# safetyFactor is used to normalize FM
    
    # CREATE DATASET LOADERS
    import quantlab.ImageNet.preprocess as pp
    datasetTrain, datasetVal, _ = pp.load_datasets(augment=False)
    if training:
        dataset = datasetTrain
        model.train()
    else:
        dataset = datasetVal
        model.eval()
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)

    # SELECT MODULES
    msReLU = list(filter(lambda m: type(m) == torch.nn.modules.ReLU or type(m) == torch.nn.modules.ReLU6, model.modules()))
    msConv = list(filter(lambda m: type(m) == torch.nn.modules.Conv2d, model.modules()))
    msBN = list(filter(lambda m: type(m) == torch.nn.modules.BatchNorm2d, model.modules()))

    #register hooks to get intermediate outputs: 
    def setupFwdHooks(modules):
      outputs = []
      def hook(module, input, output):
          outputs.append(output.detach().contiguous().clone())
      for i, m in enumerate(modules): 
        m.register_forward_hook(hook)
      return outputs

    outputsReLU = setupFwdHooks(msReLU)
    outputsConv = setupFwdHooks(msConv)
    outputsBN   = setupFwdHooks(msBN)
    
    # PASS IMAGES THROUGH NETWORK
    outputSetsMaxMulti = []
    outputSets = [outputsReLU, outputsConv, outputsBN]
    dataIterator = iter(dataLoader)
    
    (image, target) = next(dataIterator)

    if training:
        model.train()
        outp = model(image)
        loss = loss_func(outp, target)
        loss.backward()
    else: 
        model.eval()
        outp = model(image)

    tmp = [[outp.max().item() for outp in outputs] 
             for outputs in outputSets]
    outputSetsMaxMulti.append(tmp)
    
    # NORMALIZE FMs between 0 and 1
    outputSetsMax = [np.array([om2[i] for om2 in outputSetsMaxMulti]).max(axis=0) for i in range(len(outputSets))]
    for outputs, outputsMax in zip(outputSets, outputSetsMax):
      for op, opmax in zip(outputs,outputsMax):
        op.mul_(safetyFactor/opmax)

    return outputsReLU, outputsConv, outputsBN

