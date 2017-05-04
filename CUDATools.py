from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.driver as cuda
from pycuda.tools import DeviceData
from pycuda.tools import OccupancyRecord as occupancy

# Difference between int and long long for id's inside kernels take place at 2D arrays of
# length above 46341 and for 3D above 1290 point per dimension, this appears for array size of 17 GBs

def setDevice(ndev = None):
      ''' To use CUDA or OpenCL you need a context and a device to stablish the context o                   communication '''
      import pycuda.autoinit
      nDevices = cuda.Device.count()
      print "Available Devices:"
      for i in range(nDevices):
            dev = cuda.Device( i )
            print "  Device {0}: {1}".format( i, dev.name() )
      devNumber = 0
      if nDevices > 1:
            if ndev == None:
                devNumber = int(raw_input("Select device number: "))
            else:
                devNumber = ndev
      dev = cuda.Device( devNumber)
      cuda.Context.pop()  #Disable previus CUDA context
      ctxCUDA = dev.make_context()
      print "Using device {0}: {1}".format( devNumber, dev.name() )
      return ctxCUDA, dev


def getKernelInfo(kernel,nthreads, rt=True):
    ''' This function returns info about kernels theoretical performance, but warning is not trivial to optimize! '''
    shared=kernel.shared_size_bytes
    regs=kernel.num_regs
    local=kernel.local_size_bytes
    const=kernel.const_size_bytes
    mbpt=kernel.max_threads_per_block
    #threads =  #self.block_size_x* self.block_size_y* self.block_size_z
    occupy = occupancy(devdata, nthreads, shared_mem=shared, registers=regs)
    print "==Kernel Memory=="
    print("""Local:        {0}
Shared:       {1}
Registers:    {2}
Const:        {3}
Max Threads/B:{4}""".format(local,shared,regs,const,mbpt))
    print "==Occupancy=="
    print("""Blocks executed by MP: {0}
Limited by:            {1}
Warps executed by MP:  {2}
Occupancy:             {3}""".format(occupy.tb_per_mp,occupy.limited_by,occupy.warps_per_mp,occupy.occupancy))
    if rt:
        return occupy.occupancy

def gpuMesureTime(myKernel, ntimes=1000):
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    for i in range(ntimes):
      myKernel()
    end.record()
    end.synchronize()
    timeGPU = start.time_till(end)*1e-3
    print "Call the function {0} times takes in GPU {1} seconds.\n".format(ntimes,timeGPU)
    print "{0} seconds per call".format(timeGPU/ntimes)
    return timeGPU

def precisionCU(p = 'float'):
  '''(presicion) p = float,cfloat,double,cdouble'''
  if p == 'float':
    return np.float32, p, p
  if p == 'cfloat':
    return np.complex64, 'pycuda::complex<float>', p
  if p == 'double':
    return np.float64, p, p
  if p == 'cdouble':
    return np.complex128, 'pycuda::complex<double>', p

def optKernels(kFile,pres,subBlGr = False, cuB=(1,1,1), cuG=(1,1,1),compiling=False,myDir=None):

    if pres == 'float':
      cString = 'f'
      kFile = kFile.replace('cuPres', pres)
      kFile = kFile.replace('cuStr', cString)
      kFile = kFile.replace('cuName', pres) # for textures
    if pres == 'double':
      cString = ''
      kFile = kFile.replace('cuPres', pres)
      kFile = kFile.replace('cuStr', cString)
      kFile = kFile.replace('cuName', pres)
    if pres == 'cfloat':
      cString = ''
      presicion = 'pycuda::complex<float>'
      kFile = kFile.replace('cuPres', presicion)
      kFile = kFile.replace('cuStr', cString)
      kFile = kFile.replace('cuName', pres)
    if pres == 'cdouble':
      cString = ''
      presicion = 'pycuda::complex<double>'
      kFile = kFile.replace('cuPres', presicion)
      kFile = kFile.replace('cuStr', cString)
      kFile = kFile.replace('cuName', pres)

    if subBlGr:
        downVar = ['blockDim.x','blockDim.y','blockDim.z','gridDim.x','gridDim.y','gridDim.z']
        upVar      = [str(cuB[0]),str(cuB[1]),str(cuB[2]),
                      str(cuG[0]),str(cuG[1]),str(cuG[2])]
        dicVarOptim = dict(zip(downVar,upVar))
        for i in downVar:
            kFile = kFile.replace(i,dicVarOptim[i])
    if compiling:
      kFile = SourceModule(kFile,include_dirs=[myDir])
    return kFile

# La siguiente funcion permite crerar un CUDA ARRAY desde un arreglo numpy en Host ( CPU) Solo para 2 y 3 Dimensione
# Una dimansion ya esta implementada via GPUArray (ver documentacion)
# Esta ya soporta doubles de 64 bits
def np3DtoCudaArray(npArray, prec, order = "C", allowSurfaceBind=False):
  ''' Some parameters like stride are explained in PyCUDA: driver.py test_driver.py gpuarray.py'''
  # For 1D-2D Cuda Arrays the descriptor is the same just puttin LAYERED flags
#   if order != "C": raise LogicError("Just implemented for C order")
  dimension = len(npArray.shape)
  case = order in ["C","F"]
  if not case:
    raise LogicError("order must be either F or C")
#   if dimension == 1:
#       w = npArray.shape[0]
#       h, d = 0,0
  if dimension == 2:
      if order == "C": stride = 0
      if order == "F": stride = -1
      h, w = npArray.shape
      d = 1
      if allowSurfaceBind:
        descrArr = cuda.ArrayDescriptor3D()
        descrArr.width = w
        descrArr.height = h
        descrArr.depth = d
      else:
        descrArr = cuda.ArrayDescriptor()
        descrArr.width = w
        descrArr.height = h
#         descrArr.depth = d
  elif dimension == 3:
      if order == "C": stride = 1
      if order == "F": stride = 1
      d, h, w = npArray.shape
      descrArr = cuda.ArrayDescriptor3D()
      descrArr.width = w
      descrArr.height = h
      descrArr.depth = d
  else:
      raise LogicError("CUDArray dimesnsion 2 and 3 supported at the moment ... ")
  if prec == 'float':
    descrArr.format = cuda.dtype_to_array_format(npArray.dtype)
    descrArr.num_channels = 1
  elif prec == 'cfloat': # Hack for complex 64 = (float 32, float 32) == (re,im)
    descrArr.format = cuda.array_format.SIGNED_INT32 # Reading data as int2 (hi=re,lo=im) structure
    descrArr.num_channels = 2
  elif prec == 'double': # Hack for doubles
    descrArr.format = cuda.array_format.SIGNED_INT32 # Reading data as int2 (hi,lo) structure
    descrArr.num_channels = 2
  elif prec == 'cdouble': # Hack for doubles
    descrArr.format = cuda.array_format.SIGNED_INT32 # Reading data as int4 (re=(hi,lo),im=(hi,lo)) structure
    descrArr.num_channels = 4
  else:
    descrArr.format = cuda.dtype_to_array_format(npArray.dtype)
    descrArr.num_channels = 1

  if allowSurfaceBind:
    if dimension==2:  descrArr.flags |= cuda.array3d_flags.ARRAY3D_LAYERED
    descrArr.flags |= cuda.array3d_flags.SURFACE_LDST

  cudaArray = cuda.Array(descrArr)
  if allowSurfaceBind or dimension==3 :
    copy3D = cuda.Memcpy3D()
    copy3D.set_src_host(npArray)
    copy3D.set_dst_array(cudaArray)
    copy3D.width_in_bytes = copy3D.src_pitch = npArray.strides[stride]
#     if dimension==3: copy3D.width_in_bytes = copy3D.src_pitch = npArray.strides[1] #Jut C order support
#     if dimension==2: copy3D.width_in_bytes = copy3D.src_pitch = npArray.strides[0] #Jut C order support
    copy3D.src_height = copy3D.height = h
    copy3D.depth = d
    copy3D()
    return cudaArray, copy3D
  else:
#     if dimension == 3:
#       copy3D = cuda.Memcpy3D()
#       copy3D.set_src_host(npArray)
#       copy3D.set_dst_array(cudaArray)
#       copy3D.width_in_bytes = copy3D.src_pitch = npArray.strides[stride]
# #       if dimension==3: copy3D.width_in_bytes = copy3D.src_pitch = npArray.strides[1] #Jut C order support
# #       if dimension==2: copy3D.width_in_bytes = copy3D.src_pitch = npArray.strides[0] #Jut C order support
#       copy3D.src_height = copy3D.height = h
#       copy3D.depth = d
#       copy3D()
#       return cudaArray, copy3D
#     if dimension == 2:
      cudaArray = cuda.Array(descrArr)
      copy2D = cuda.Memcpy2D()
      copy2D.set_src_host(npArray)
      copy2D.set_dst_array(cudaArray)
      copy2D.width_in_bytes = copy2D.src_pitch = npArray.strides[stride]
#       copy2D.width_in_bytes = copy2D.src_pitch = npArray.strides[0] #Jut C order support
      copy2D.src_height = copy2D.height = h
      copy2D(aligned=True)
      return cudaArray, copy2D

def getFreeMemory(show=True):
    ''' Return the free memory of the device,. Very usful to look for save device memory '''
    Mb = 1024.*1024.
    Mbytes = float(cuda.mem_get_info()[0])/Mb
    if show:
      print "Free Global Memory: %f Mbytes" %Mbytes

    return cuda.mem_get_info()[0]/Mb

ctx,device = setDevice()
devdata = DeviceData(device)
