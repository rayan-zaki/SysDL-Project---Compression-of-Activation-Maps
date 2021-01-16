#Libraries
import math
import numpy as np
from bin_dec_quant import valuesToBinary, binary2dec, gbits

#Compression Algorithms
#-----------------------
#--------------------------------------------------------------------------
#Zero Value Coding - Compression

def ZVC(values, wordwidth=None, debug=False):
    symbols = ''.join(['0' if v == 0 else '1' + valuesToBinary(v, wordwidth=wordwidth) for v in values])
    return ''.join(symbols)
    
#--------------------------------------------------------------------------
#Zero Value Coding - Decompression
    
def ZVDecompress(codedStream,wordwidth):
    decodedList = []
    i=0
    while i<len(codedStream):
        if codedStream[i]=='0':
            decodedList.append(0)
            i=i+1
        elif codedStream[i]=='1':
            decodedList.append(binary2dec(codedStream[i+1:i+1+wordwidth]))
            i=i+1+wordwidth
    return decodedList

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#Zero Run Length Encoding - Compression

def zeroRLE(values, maxZeroRun=63, wordwidth=None):
    if maxZeroRun == None:
        maxZeroRun = len(values)
    maxZeroRunBits = math.ceil(math.log2(maxZeroRun+1))

    codedStream = ''
    zero_count = 0
    for v in values:
        if v == 0 and zero_count < maxZeroRun:
            zero_count += 1
        else:
            if zero_count > 0:
                #flush zeros
                codedStream += '0' + bin(zero_count)[2:].zfill(maxZeroRunBits)
            #keep track of current element
            if v == 0:
                zero_count = 1
            else:
                codedStream += '1' + valuesToBinary(v, wordwidth=wordwidth)
                zero_count = 0
    if zero_count > 0: # if after coming out of loop zero_count is non zero
        codedStream += '0' + bin(zero_count)[2:].zfill(maxZeroRunBits)

    return codedStream
#-------------------------------------------------------------------------------
#Zero Run Length Encoding - Decompression

def zeroRLDecompress(codedStream, maxZeroRun=63, wordwidth=None, dtype=np.int8):
    if maxZeroRun == None:
        maxZeroRun = len(values)
    
    wordwidthInput = dtype(0).itemsize*8 #dtype.nbytes*8
        
    if wordwidth is None:
        wordwidth = wordwidthInput
    maxZeroRunBits = math.ceil(math.log2(maxZeroRun+1))
    decodedList = []
    i=0
    while i<len(codedStream):
        if codedStream[i]=='0':
            #print(i)
            decodedList.extend(np.zeros(int(codedStream[i+1:i+1+maxZeroRunBits],2),dtype=dtype))
            i=i+1+maxZeroRunBits
        elif codedStream[i]=='1':
            decodedList.append(binary2dec(codedStream[i+1:i+1+wordwidth]))
            i=i+1+wordwidth
    return decodedList

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Grid Based Run Length Encoding with outlier - Compression
def GRLE_wo(Array, tileSize=4 ,maxZeroRun=63):
    if len(Array.shape)<4:
        Array=np.array(Array,ndmin=4)
    elif len(Array.shape)>4:
        assert(False)
    assert(Array.dtype==np.int8)
    if maxZeroRun == None:
        maxZeroRun = Array.shape[1]*Array.shape[2]*Array.shape[3]
    maxZeroRunBits = math.ceil(math.log2(maxZeroRun+1))
    ww4 = 4
    ww8 = 8
    
    codedStream = ''
    zero_count = 0
    for b in range(Array.shape[0]):
        for k in range(Array.shape[1]):
            i=0
            while i<Array.shape[2] :
                j=0
                while j<Array.shape[3]:
                    values=Array[b][k][i:i+tileSize,j:j+tileSize].flatten()
                    for v in values:
                        if v == 0 and zero_count < maxZeroRun:
                            zero_count += 1
                        else:
                            if zero_count > 0:
                                #append zero stream
                                codedStream += '00' + bin(zero_count)[2:].zfill(maxZeroRunBits)
                                
                            #current element
                            if v == 0:
                                zero_count = 1
                            else:
                                if gbits(v)<=4:
                                    codedStream += '01' + valuesToBinary(v, wordwidth=4)
                                else:
                                    codedStream += '10' + valuesToBinary(v, wordwidth=8)
                                zero_count = 0
                    j+=tileSize
                i+=tileSize
        if zero_count > 0:   # if after coming out of image loop zero_count is non zero 
            #append zero stream before next channel
            codedStream += '00' + bin(zero_count)[2:].zfill(maxZeroRunBits)
            zero_count = 0

    return codedStream

#-------------------------------------------------------------------------------
# Grid Based Run Length Encoding with outliers- Decompression
def GRLDecompress_wo(codedStream, maxZeroRun=63):
    # Note that default maxZeroRun=63 implies 6 bits to store zero run
    # this is important parameter to tweek because mostly there might not be such big continuous zero run
    ww4 = 4
    ww8 = 8
    maxZeroRunBits = math.ceil(math.log2(maxZeroRun+1))
    decodedList = []
    i=0
    while i<len(codedStream):
        if codedStream[i:i+2]=='00':
            decodedList.extend(np.zeros(int(codedStream[i+2:i+2+maxZeroRunBits],2),dtype=dtype))
            i=i+2+maxZeroRunBits
        elif codedStream[i:i+2]=='01':
            decodedList.append(binary2dec(codedStream[i+2:i+2+ww4]))
            i=i+2+ww4
        elif codedStream[i:i+2]=='10':
            decodedList.append(binary2dec(codedStream[i+1:i+1+ww8]))
            i=i+2+ww8
    return decodedList


