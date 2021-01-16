#Libraries
import numpy

def assertZVD(arr,decodedList):
    sh=arr.shape
    dtyp=arr[0].dtype
    decoded_arr=np.array(decodedList).astype(dtyp).reshape(sh)
    assert(qa.all()==decoded_arr.all())
    return

def assertzeroRLD(arr, decodedList):
    sh=arr.shape
    dtyp=arr[0].dtype
    decoded_arr=np.array(decodedList).astype(dtyp).reshape(sh)
    assert(qa.all()==decoded_arr.all())
    return

def assertGRLD(arr, decodedList,tileSize):
    if len(arr.shape)<4:
        arr=np.array(arr,ndmin=4)
    elif len(arr.shape)>4:
        assert(False)
    B=arr.shape[0]
    C=arr.shape[1]
    H=arr.shape[2]
    W=arr.shape[3]
    dtyp=arr[0].dtype
    decoded_cont=np.array(decodedList).astype(dtyp)
    decoded_arr=np.zeros((B,C,H,W))
    ctr=0
    for b in range(B):
        for k in range(C):
            i=0
            while i<H :
                j=0
                while j<W:
                    t_sh=decoded_arr[b][k][i:i+tileSize,j:j+tileSize].shape
                    t_sz=decoded_arr[b][k][i:i+tileSize,j:j+tileSize].size
                    decoded_arr[b][k][i:i+tileSize,j:j+tileSize] = decoded_cont[ctr:ctr+t_sz].reshape(t_sh)
                    ctr+=t_sz
                    j+=tileSize
                i+=tileSize
    assert(qa.all()==decoded_arr.all())
    return
