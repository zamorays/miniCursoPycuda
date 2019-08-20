#include <pycuda-complex.hpp>
#include <pycuda-helpers.hpp>
#include <surface_functions.h>

#define pi 3.14159265
#define phi 1.6180339

typedef  pycuda::complex<cudaPres> pyComplex;

extern "C++" {
typedef float fp_tex_float;
typedef int2 fp_tex_double;
typedef uint2 fp_tex_cfloat;
typedef int4 fp_tex_cdouble;
    
  __device__ void fp_surf2Dwrite(double var,surface<void, cudaSurfaceType2D> surf, int i, int j, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_double auxvar;
    auxvar.x =  __double2loint(var);
    auxvar.y =  __double2hiint(var);
    surf2Dwrite(auxvar, surf, i*sizeof(fp_tex_double), j, mode);
  }

  __device__ void fp_surf2Dwrite(pycuda::complex<float> var,surface<void, cudaSurfaceType2D> surf, int i, int j, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cfloat auxvar;
    auxvar.x =  __float_as_int(var._M_re);
    auxvar.y =  __float_as_int(var._M_im);
    surf2Dwrite(auxvar, surf, i*sizeof(fp_tex_cfloat), j,mode);
  }

  __device__ void fp_surf2Dwrite(pycuda::complex<double> var,surface<void, cudaSurfaceType2D> surf, int i, int j, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cdouble auxvar;
    auxvar.x =  __double2loint(var._M_re);
    auxvar.y =  __double2hiint(var._M_re);

    auxvar.z = __double2loint(var._M_im);
    auxvar.w = __double2hiint(var._M_im);
    surf2Dwrite(auxvar, surf, i*sizeof(fp_tex_cdouble), j,mode);
  }
   
   __device__ void fp_surf2Dread(double *var, surface<void, cudaSurfaceType2D> surf, int i, int j, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_double v;
    surf2Dread(&v, surf, i*sizeof(fp_tex_double), j, mode);
    *var = __hiloint2double(v.y, v.x);
  }

  __device__ void fp_surf2Dread(pycuda::complex<float> *var, surface<void, cudaSurfaceType2D> surf, int i, int j, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cfloat v;
    surf2Dread(&v, surf, i*sizeof(fp_tex_cfloat), j, mode);
    *var = pycuda::complex<float>(__int_as_float(v.x), __int_as_float(v.y));
  }

  __device__ void fp_surf2Dread(pycuda::complex<double> *var, surface<void, cudaSurfaceType2D> surf, int i, int j, enum cudaSurfaceBoundaryMode mode)
  {
    fp_tex_cdouble v;
    surf2Dread(&v, surf, i*sizeof(fp_tex_cdouble), j, mode);
    *var = pycuda::complex<double>(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }
    
}

surface< void, cudaSurfaceType3D> surf_psi ;
texture< fp_pres, cudaTextureType3D, cudaReadModeElementType> tex_psi ;

surface< void, cudaSurfaceType2DLayered> surf_psi2D;
surface< void, cudaSurfaceType2D> surf_psi2DNL;
texture< fp_pres, cudaTextureType2D, cudaReadModeElementType> tex_psi2D ;


__device__ cudaPres KspaceFFT(int tid, int nPoint, cudaPres L){
cudaPres Kfft;
if (tid < nPoint/2){
    Kfft = 2.0f*pi*(tid)/L;
}
else {
    Kfft = 2.0f*pi*(tid-nPoint)/L;
}
return Kfft;
}

__global__ void gaussian_kernel( cudaPres dx,cudaPres dy, cudaPres dz,
                cudaPres xMin,cudaPres yMin, cudaPres zMin,
                cudaPres a,cudaPres b, cudaPres c,
                cudaPres d,cudaPres e, cudaPres f, int caso,
                pyComplex *psi){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int t_k = blockIdx.z*blockDim.z + threadIdx.z;
int tid   = gridDim.z * blockDim.z * gridDim.y * blockDim.y * t_i + gridDim.z * blockDim.z * t_j + t_k;
cudaPres x=xMin+t_i*dx;
cudaPres y=yMin+t_j*dy;
cudaPres z=zMin+t_k*dz;
pyComplex value;

if (caso==0){
value._M_re=exp(-a*x*x-b*y*y-c*z*z);
value._M_im=exp(-d*x*x-e*y*y-f*z*z);
}
if (caso==1){
value._M_re=x;
value._M_im=y;
}
psi[tid] = value;
}


__global__ void laplaFFT_kernel( cudaPres Lx,cudaPres Ly, cudaPres Lz,
int nPointX,int nPointY, int nPointZ,
pyComplex *fftTrnf){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int t_k = blockIdx.z*blockDim.z + threadIdx.z;
int tid   = gridDim.z * blockDim.z * gridDim.y * blockDim.y * t_i + gridDim.z * blockDim.z * t_j + t_k;

cudaPres kX = KspaceFFT(t_i,nPointX, Lx);//kx[t_j];
cudaPres kY = KspaceFFT(t_j,nPointY, Ly);//ky[t_i];
cudaPres kZ = KspaceFFT(t_k,nPointZ, Lz);//kz[t_k];
cudaPres k2 = kX*kX + kY*kY + kZ*kZ;
pyComplex value = fftTrnf[tid];
fftTrnf[tid] = -k2*value;
}

__global__ void laplaFDtex_kernel(cudaPres dx, cudaPres dy, cudaPres dz, pyComplex *func_d){
int t_x = blockIdx.x*blockDim.x + threadIdx.x;
int t_y = blockIdx.y*blockDim.y + threadIdx.y;
int t_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid = gridDim.z * blockDim.z * gridDim.y * blockDim.y * t_x + blockDim.z * gridDim.z * t_y + t_z;

pyComplex center, right, left, up, down, top, bottom;

center = fp_tex3D(tex_psi, t_z,   t_y,   t_x);
up =     fp_tex3D(tex_psi, t_z,   t_y+1, t_x);
down =   fp_tex3D(tex_psi, t_z,   t_y-1, t_x);
right =  fp_tex3D(tex_psi, t_z, t_y,   t_x+1);
left =   fp_tex3D(tex_psi, t_z, t_y,   t_x-1);
top =    fp_tex3D(tex_psi, t_z+1,   t_y,   t_x);
bottom = fp_tex3D(tex_psi, t_z-1,   t_y,   t_x);

cudaPres  drInv = 1.0/dy;
pyComplex laplacian = (up + down - 2.0cString*center )*drInv*drInv;
drInv = 1.0/dx;
laplacian += (right + left - 2.0cString*center )*drInv*drInv;
drInv = 1.0/dz;
laplacian += (top + bottom - 2.0cString*center )*drInv*drInv;
func_d[tid] = laplacian;
}

__global__ void laplaFDsurf_kernel(cudaPres dx, cudaPres dy, cudaPres dz, pyComplex *func_d){
int t_x = blockIdx.x*blockDim.x + threadIdx.x;
int t_y = blockIdx.y*blockDim.y + threadIdx.y;
int t_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid = gridDim.z * blockDim.z * gridDim.y * blockDim.y * t_x + blockDim.z * gridDim.z * t_y + t_z;

pyComplex center, right, left, up, down, top, bottom;

fp_surf3Dread(&center,surf_psi, t_z,   t_y,   t_x, cudaBoundaryModeZero);
fp_surf3Dread(&up,surf_psi, t_z,   t_y+1, t_x, cudaBoundaryModeZero);
fp_surf3Dread(&down,surf_psi, t_z,   t_y-1, t_x, cudaBoundaryModeZero);
fp_surf3Dread(&right,surf_psi, t_z, t_y,   t_x+1, cudaBoundaryModeZero);
fp_surf3Dread(&left,surf_psi, t_z, t_y,   t_x-1, cudaBoundaryModeZero);
fp_surf3Dread(&top,surf_psi, t_z+1,   t_y,   t_x, cudaBoundaryModeZero);
fp_surf3Dread(&bottom,surf_psi, t_z-1,   t_y,   t_x, cudaBoundaryModeZero);

cudaPres drInv = 1.0/dy;
pyComplex laplacian = (up + down - 2.0cString*center )*drInv*drInv;
drInv = 1.0/dx;
laplacian += (right + left - 2.0cString*center )*drInv*drInv;
drInv = 1.0/dz;
laplacian += (top + bottom - 2.0cString*center )*drInv*drInv;
func_d[tid] = laplacian;
}

__global__ void test_tex_kernel( pyComplex *func_d){
int t_x = blockIdx.x*blockDim.x + threadIdx.x;
int t_y = blockIdx.y*blockDim.y + threadIdx.y;
int t_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid = gridDim.z * blockDim.z * gridDim.y * blockDim.y * t_x + blockDim.z * gridDim.z * t_y + t_z;

pyComplex center;

center = fp_tex3D(tex_psi, t_z,   t_y,   t_x);

func_d[tid] = center;
}

__global__ void test_surf_kernel(pyComplex *func_d){
int t_x = blockIdx.x*blockDim.x + threadIdx.x;
int t_y = blockIdx.y*blockDim.y + threadIdx.y;
int t_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid = gridDim.z * blockDim.z * gridDim.y * blockDim.y * t_x + blockDim.z * gridDim.z * t_y + t_z;

pyComplex center;

fp_surf3Dread(&center,surf_psi, t_z,   t_y,   t_x, cudaBoundaryModeZero);

func_d[tid] = center;
}

__global__ void setzero_kernel(pyComplex *func_d){

int t_x = blockIdx.x*blockDim.x + threadIdx.x;
int t_y = blockIdx.y*blockDim.y + threadIdx.y;
int t_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid = gridDim.z * blockDim.z * gridDim.y * blockDim.y * t_x + blockDim.z * gridDim.z * t_y + t_z;
func_d[tid] *=0;
}

__global__ void write2surf_kernel(pyComplex *realArray1){

// This kernel writes quantum pressure and non linear term of energy

int t_x = blockIdx.x*blockDim.x + threadIdx.x;
int t_y = blockIdx.y*blockDim.y + threadIdx.y;
int t_z = blockIdx.z*blockDim.z + threadIdx.z;
int tid = gridDim.z * blockDim.z * gridDim.y * blockDim.y * t_x + blockDim.z * gridDim.z * t_y + t_z;

pyComplex arr1 = realArray1[tid];
// cudaPres arr2 = realArray2[tid];

// Write to Surfaces

fp_surf3Dwrite(  arr1, surf_psi,   t_z, t_y, t_x,  cudaBoundaryModeClamp);
// fp_surf3Dwrite(  arr2, surf_psi0OutImag,  t_x*sizeof(cudaPres), t_y, t_z,  cudaBoundaryModeClamp);
}


//#################################################  2D


__global__ void gaussian_kernel2D( cudaPres dx,cudaPres dy, 
                cudaPres xMin,cudaPres yMin,
                cudaPres a,cudaPres b,
                cudaPres d,cudaPres e, int caso,
                pyComplex *psi){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;
cudaPres x=xMin+t_i*dx;
cudaPres y=yMin+t_j*dy;

pyComplex value;

if (caso==0){
value._M_re=exp(-a*x*x-b*y*y);
value._M_im=exp(-d*x*x-e*y*y);
}
if (caso==1){
value._M_re=x;
value._M_im=y;
}
psi[tid] = value;
}


__global__ void laplaFFT_kernel2D( cudaPres Lx,cudaPres Ly, 
int nPointX,int nPointY, 
pyComplex *fftTrnf){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;

cudaPres kX = KspaceFFT(t_i,nPointX, Lx);//kx[t_j];
cudaPres kY = KspaceFFT(t_j,nPointY, Ly);//ky[t_i];
//cudaPres kZ = KspaceFFT(t_k,nPointZ, Lz);//kz[t_k];
cudaPres k2 = kX*kX + kY*kY ;
pyComplex value = fftTrnf[tid];
fftTrnf[tid] = -k2*value;
}

__global__ void laplaFDtex_kernel2D(cudaPres dx, cudaPres dy, pyComplex *func_d){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;

pyComplex center, right, left, up, down;

center = fp_tex2D(tex_psi2D,  t_j,   t_i);
right =  fp_tex2D(tex_psi2D,  t_j,   t_i+1);
left =   fp_tex2D(tex_psi2D,  t_j,   t_i-1);
up =     fp_tex2D(tex_psi2D,  t_j+1, t_i);
down =   fp_tex2D(tex_psi2D,  t_j-1, t_i);

cudaPres  drInv = 1.0/dy;
pyComplex laplacian = (up + down - 2.0cString*center )*drInv*drInv;
drInv = 1.0/dx;
laplacian += (right + left - 2.0cString*center )*drInv*drInv;
//drInv = 1.0/dz;
//laplacian += (top + bottom - 2.0cString*center )*drInv*drInv;
func_d[tid] = laplacian;
}

__global__ void laplaFDsurf_kernel2D(cudaPres dx, cudaPres dy,  pyComplex *func_d){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;

pyComplex center, right, left, up, down ;

fp_surf2DLayeredread(&center,surf_psi2D,    t_j,   t_i, int(0), cudaBoundaryModeZero);
fp_surf2DLayeredread(&up,    surf_psi2D,    t_j+1, t_i, int(0), cudaBoundaryModeZero);
fp_surf2DLayeredread(&down,  surf_psi2D,    t_j-1, t_i, int(0), cudaBoundaryModeZero);
fp_surf2DLayeredread(&right, surf_psi2D,    t_j,   t_i+1, int(0), cudaBoundaryModeZero);
fp_surf2DLayeredread(&left,  surf_psi2D,    t_j,   t_i-1, int(0), cudaBoundaryModeZero);
//fp_surf3Dread(&top,surf_psi, t_z+1,   t_y,   t_x, cudaBoundaryModeZero);
//fp_surf3Dread(&bottom,surf_psi, t_z-1,   t_y,   t_x, cudaBoundaryModeZero);

cudaPres drInv = 1.0/dy;
pyComplex laplacian = (up + down - 2.0cString*center )*drInv*drInv;
drInv = 1.0/dx;
laplacian += (right + left - 2.0cString*center )*drInv*drInv;
func_d[tid] = laplacian;
}

__global__ void laplaFDsurf_kernel2DNL(cudaPres dx, cudaPres dy,  pyComplex *func_d){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;

pyComplex center, right, left, up, down ;

fp_surf2Dread(&center,surf_psi2DNL,    t_j,   t_i, cudaBoundaryModeZero);
fp_surf2Dread(&up,    surf_psi2DNL,    t_j+1, t_i,  cudaBoundaryModeZero);
fp_surf2Dread(&down,  surf_psi2DNL,    t_j-1, t_i,  cudaBoundaryModeZero);
fp_surf2Dread(&right, surf_psi2DNL,    t_j,   t_i+1,  cudaBoundaryModeZero);
fp_surf2Dread(&left,  surf_psi2DNL,    t_j,   t_i-1,  cudaBoundaryModeZero);
//fp_surf3Dread(&top,surf_psi, t_z+1,   t_y,   t_x, cudaBoundaryModeZero);
//fp_surf3Dread(&bottom,surf_psi, t_z-1,   t_y,   t_x, cudaBoundaryModeZero);

cudaPres drInv = 1.0/dy;
pyComplex laplacian = (up + down - 2.0cString*center )*drInv*drInv;
drInv = 1.0/dx;
laplacian += (right + left - 2.0cString*center )*drInv*drInv;
func_d[tid] = laplacian;
}

__global__ void test_tex_kernel2D( pyComplex *func_d){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;
    
pyComplex center;

center = fp_tex2D(tex_psi2D,  t_j,   t_i);

func_d[tid] = center;
}

__global__ void test_surf_kernel2D(pyComplex *func_d){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;
    
pyComplex center;

fp_surf2DLayeredread(&center,surf_psi2D,  t_j,   t_i, int(0), cudaBoundaryModeZero);

func_d[tid] = center;
}

__global__ void test_surf_kernel2DNL(pyComplex *func_d){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;
    
pyComplex center;

fp_surf2Dread(&center,surf_psi2DNL,  t_j,   t_i, cudaBoundaryModeZero);

func_d[tid] = center;
}

__global__ void setzero_kernel2D(pyComplex *func_d){

int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;
func_d[tid] *=0;
}

__global__ void write2surf_kernel2D(pyComplex *realArray1){

// This kernel writes quantum pressure and non linear term of energy

int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;

pyComplex arr1 = realArray1[tid];
// cudaPres arr2 = realArray2[tid];

// Write to Surfaces

fp_surf2DLayeredwrite(  arr1, surf_psi2D,   t_j, t_i, int(0),  cudaBoundaryModeClamp);
// fp_surf3Dwrite(  arr2, surf_psi0OutImag,  t_x*sizeof(cudaPres), t_y, t_z,  cudaBoundaryModeClamp);
}

__global__ void write2surf_kernel2DNL(pyComplex *realArray1){

// This kernel writes quantum pressure and non linear term of energy

int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid   = gridDim.y * blockDim.y * t_i +t_j ;

pyComplex arr1 = realArray1[tid];
// cudaPres arr2 = realArray2[tid];

// Write to Surfaces

fp_surf2Dwrite(  arr1, surf_psi2DNL,   t_j, t_i,  cudaBoundaryModeClamp);
// fp_surf3Dwrite(  arr2, surf_psi0OutImag,  t_x*sizeof(cudaPres), t_y, t_z,  cudaBoundaryModeClamp);
}