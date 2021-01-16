import functools
import numpy as np
from bin_dec_quant import quantize

class Analyzer:
    def __init__(self, quantMethod, compressor):
        self.quantMethod_default = quantMethod
        self.compressor_default = compressor
    
    @functools.lru_cache(maxsize=8*200*100)
    def ComprStats(self, t, quant=None, compressor=None, flat=True):
      if quant == None:
        quant = self.quantMethod_default
      if compressor == None: 
        compressor = self.compressor_default
        
      t, numBit, dtype = quantize(t, quant=quant)
      if flat==True:
          valueStream = t.view(-1).numpy().astype(dtype) #Flatten the tensor as valuestream of data type: dtype
          comprSizeBaseline = len(valueStream)*numBit
      else:
          valueStream = t.numpy().astype(dtype) #Dont Flatten for GRLE
          comprSizeBaseline = len(valueStream.flatten())*numBit

      codeStream = compressor(valueStream) #pass all the values to the compressor and it return a codestream
      comprSize = len(codeStream) #strmLen
      comprRatio = comprSizeBaseline/len(codeStream) #strmLen
      return comprRatio, comprSize, comprSizeBaseline

    def getComprRatio(self, t, quant=None, compressor=None):
      if quant == None:
        quant = self.quantMethod_default
      if compressor == None: 
        compressor = self.compressor_default

      comprRatio, _, _ = self.ComprStats(t, quant=quant, compressor=compressor)
      return comprRatio

    def getSparsity(self, t, quant=None):
      if quant == None:
        quant = self.quantMethod_default
        
      return t.contiguous().view(-1).eq(0).long().sum().item()/t.numel()

    def TotalComprStats(self, outputs, quant=None, compressor=None):
      comprProps = [self.ComprStats(outp, quant=quant, compressor=compressor) 
                    for outp in outputs]
      totalLen = np.array([l for _, l, _ in comprProps]).sum()
      uncomprLen = np.array([l for _, _, l in comprProps]).sum()
      return uncomprLen/totalLen, totalLen, uncomprLen
