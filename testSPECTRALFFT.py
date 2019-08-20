
from pyfft.cuda import Plan as plan1
from skcuda.fft import Plan as plan2
from skcuda.fft import fft as skfft
from skcuda.fft import ifft as iskfft
from reikna.fft import FFT
from reikna.cluda import cuda_api
from CUDATools import *

ctx,device,devdata = setDevice(ndev=0)
getFreeMemory()

prec='single'

src = open('spectralTest.cu','r')
auxKernels = src.read()
if prec=='single':
    precf = np.float32
    precc = np.complex64
    kernelprec = 'float'
    divisionTol = precf(1e-9)
    cString = 'f'
    fpName = 'fp_tex_cfloat'
elif prec=='double':
    precf = np.float64
    precc = np.complex128
    kernelprec = 'double'
    divisionTol = 0.0
    cString = ''
    fpName = 'fp_tex_cdouble'
auxKernels = auxKernels.replace('cudaPres', kernelprec)
auxKernels = auxKernels.replace('cString', cString)
auxKernels = auxKernels.replace('fp_pres', fpName)
compiledK = SourceModule(auxKernels)

laplaTEX =compiledK.get_function('laplaFDtex_kernel')
laplaSURF=compiledK.get_function('laplaFDsurf_kernel')
laplaFFT =compiledK.get_function('laplaFFT_kernel')
tex = compiledK.get_texref('tex_psi')
surf= compiledK.get_surfref('surf_psi')
ttex =compiledK.get_function('test_tex_kernel')
tsurf =compiledK.get_function('test_surf_kernel')
setZero =compiledK.get_function('setzero_kernel')
w2surf = compiledK.get_function('write2surf_kernel')
gaussgpu = compiledK.get_function('gaussian_kernel')

def sol3D(drs,NPoints,params):
    dr = np.array(drs,dtype=precf)
    nPoints = np.array(NPoints,dtype=np.int32)
    sideX = precf(dr[0]*(nPoints[0]-1))
    sideY = precf(dr[1]*(nPoints[1]-1))
    sideZ = precf(dr[2]*(nPoints[2]-1))

    xMax,xMin = sideX/precf(2.), -sideX/precf(2.)
    yMax,yMin = sideY/precf(2.), -sideY/precf(2.)
    zMax,zMin = sideZ/precf(2.), -sideZ/precf(2.)

    xPoints = np.array([xMin+dr[0]*i for i in range(nPoints[0])],dtype=precf)
    yPoints = np.array([yMin+dr[1]*i for i in range(nPoints[1])],dtype=precf)
    zPoints = np.array([zMin+dr[2]*i for i in range(nPoints[2])],dtype=precf)

    X, Y, Z = np.meshgrid(xPoints,yPoints,zPoints,indexing='ij')
    
    block_size_x, block_size_y, block_size_z = 8,8,8
    gridx = np.int32(nPoints[0] // block_size_x + 1 * ( nPoints[0] % block_size_x != 0 ))
    gridy = np.int32(nPoints[1] // block_size_y + 1 * ( nPoints[1] % block_size_y != 0 ))
    gridz = np.int32(nPoints[2] // block_size_z + 1 * ( nPoints[2] % block_size_z != 0 ))
    grid3D = (int(gridx), int(gridy), int(gridz))
    block3D = (int(block_size_x), int(block_size_y), int(block_size_z))
    
    a,b,c = params[0:3]
    d,e,f = params[3:6]
    
    Lf_cpu = (4*a*a*X*X+4*b*b*Y*Y+4*c*c*Z*Z-2*(a+b+c))*np.exp(-a*X*X-b*Y*Y-c*Z*Z)+(4*d*d*X*X+4*e*e*Y*Y+4*f*f*Z*Z-2*(d+e+f))*1j*np.exp(-d*X*X-e*Y*Y-f*Z*Z)
    del X,Y,Z
    return Lf_cpu,dr,[xMin,yMin,zMin],[sideX,sideY,sideZ],grid3D,block3D

def solByTex():
    laplaTEX(dR[0],dR[1],dR[2],aux_gpu,block=block3d,grid=grid3d)

def solBySurf():
    surf.set_array(farray_gpu)
    w2surf(func_gpu,block=block3d,grid=grid3d)
    laplaSURF(dR[0],dR[1],dR[2],aux_gpu,block=block3d,grid=grid3d)
    
def solByPyFft():
    myplan0.execute(func_gpu,aux_gpu)
    laplaFFT(boxSide[0],boxSide[1],boxSide[2],nxyz[0],nxyz[1],nxyz[2],
        aux_gpu,block=block3d,grid=grid3d)
    #myplan0.execute(aux_gpu,inverse=True)
    myplan0.execute(aux_gpu,aux2_gpu,inverse=True)
    
def solBySci():
    skfft(func_gpu, aux_gpu,myplan1)
    laplaFFT(boxSide[0],boxSide[1],boxSide[2],nxyz[0],nxyz[1],nxyz[2],
        aux_gpu,block=block3d,grid=grid3d)
    iskfft(aux_gpu, aux2_gpu,myplan1)
    
def solByReik():
    ret = reikFFT(aux_gpu, func_gpu )
    laplaFFT(boxSide[0],boxSide[1],boxSide[2],nxyz[0],nxyz[1],nxyz[2],
             aux_gpu,block=block3d,grid=grid3d)
    ret1 = reikFFT(aux2_gpu,aux_gpu,inverse=1)
    
if __name__ == '__main__':
    dxyz = [0.1,0.25,0.3]
    nxyz = np.array([512,256,128],dtype=np.int32)
    cpuSol,dR,boxMin,boxSide,grid3d,block3d = sol3D(dxyz,nxyz,[0.5, 0.25, 0.1, 2.0, 1.0, 0.75])
    a,b,c,d,e,f = [0.5, 0.25, 0.1, 2.0, 1.0, 0.75]
    
    imem = getFreeMemory(show=False)
    func_gpu = gpuarray.zeros(cpuSol.shape,dtype=precc)
    aux_gpu = gpuarray.zeros_like(func_gpu)
    aux2_gpu = gpuarray.zeros_like(func_gpu)
    gaussgpu(dR[0],dR[1],dR[2],boxMin[0],boxMin[1],boxMin[2],
             precf(a),precf(b),precf(c),
             precf(d),precf(e),precf(f), np.int32(0),func_gpu,block=block3d,grid=grid3d)
    farray_gpu = cuda.gpuarray_to_array(func_gpu, order='C',allowSurfaceBind=True)
    print 'GPU Arrays use: ',imem-getFreeMemory(show=False),'MB \n'
    print 'Each array costs(we use 4): ', func_gpu.nbytes/(1024.*1024.),'MB \n'
    
    imem = getFreeMemory(show=False)
    tex.set_array(farray_gpu)
    gpuMesureTime(solByTex, ntimes=100)
    solTEX = aux_gpu.get()
    print 'Texture error: ', np.sum(abs(cpuSol.real-solTEX.real)), np.sum(abs(cpuSol.imag-solTEX.imag))
    print 'Extra memory use:',imem-getFreeMemory(show=False),'MB \n'

    imem = getFreeMemory(show=False)
    setZero(aux_gpu,block=block3d,grid=grid3d)
    gpuMesureTime(solBySurf, ntimes=100)
    solSURF = aux_gpu.get()
    print 'Surface error: ', np.sum(abs(cpuSol.real-solSURF.real)), np.sum(abs(cpuSol.imag-solSURF.imag))
    print 'Extra memory use:',imem-getFreeMemory(show=False),'MB \n'
    
    imem = getFreeMemory(show=False)
    setZero(aux_gpu,block=block3d,grid=grid3d)
    setZero(aux2_gpu,block=block3d,grid=grid3d)
    myplan0 = plan1((nxyz[0],nxyz[1],nxyz[2]), dtype=precc, context=ctx)
    gpuMesureTime(solByPyFft, ntimes=100)
    solPyFFT = aux2_gpu.get()
    print 'PyFFT error: ', np.sum(abs(cpuSol.real-solPyFFT.real)), np.sum(abs(cpuSol.imag-solPyFFT.imag))
    print 'Extra memory use:',imem-getFreeMemory(show=False),'MB \n'
    #print np.sum(cpuSol.real)
    
    imem = getFreeMemory(show=False)
    setZero(aux_gpu,block=block3d,grid=grid3d)
    setZero(aux2_gpu,block=block3d,grid=grid3d)
    myplan1 = plan2(aux_gpu.shape,aux_gpu.dtype,aux_gpu.dtype)
    gpuMesureTime(solBySci, ntimes=100)
    solSci = aux2_gpu.get()/float(cpuSol.size)
    print 'SciKits error: ', np.sum(abs(cpuSol.real-solSci.real)), np.sum(abs(cpuSol.imag-solSci.imag))
    print 'Extra memory use:',imem-getFreeMemory(show=False),'MB \n'
    
    imem = getFreeMemory(show=False)
    setZero(aux_gpu,block=block3d,grid=grid3d)
    setZero(aux2_gpu,block=block3d,grid=grid3d)
    api = cuda_api()
    thr = api.Thread(ctx)
    fftPlan3 = FFT(func_gpu)
    reikFFT = fftPlan3.compile(thr)
    gpuMesureTime(solByReik, ntimes=100)
    solReik = aux2_gpu.get()
    print 'Reikna error: ', np.sum(abs(cpuSol.real-solReik.real)), np.sum(abs(cpuSol.imag-solReik.imag))
    print 'Extra memory use:',imem-getFreeMemory(show=False),'MB \n'
    
    #print np.sum(cpuSol.real),np.sum(abs(cpuSol.real))
    
    ctx.detach()