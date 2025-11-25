// ============================================================================
// sciara_fv2_cuda.cu
// CUDA GLOBAL implementation (no shared memory, no atomicAdd(double))
// Compatible with GTX 9xx (Compute Capability 5.x)
// ----------------------------------------------------------------------------
// Includes ONLY:
//   - kernels CUDA
//   - wrapper functions
//
// The main simulation loop stays in sciara_fv2.cpp
// ============================================================================

#include "Sciara.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// ----------------------------------------------------------------------------
// Macros identical to serial/OpenMP version
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j)       ((M)[(((i) * (columns)) + (j))])

#define BUF_SET(M, rows, columns, n, i, j, value) \
    ((M)[ ( (n)*(rows)*(columns) + (i)*(columns) + (j) ) ] = (value))

#define BUF_GET(M, rows, columns, n, i, j) \
    ((M)[ ( (n)*(rows)*(columns) + (i)*(columns) + (j) ) ])

// ----------------------------------------------------------------------------
// Compute global index (i,j) helper
// ----------------------------------------------------------------------------
__device__ inline bool computeIndex(int &i, int &j, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return false;

    i = y;
    j = x;
    return true;
}

// ============================================================================
// KERNELS
// ============================================================================

// ----------------------------------------------------------------------------
// emitLava
// ----------------------------------------------------------------------------
__global__ void emitLavaKernel(
    double *Sh,
    double *Sh_next,
    double *ST_next,
    int rows, int cols,
    const int *ventX,
    const int *ventY,
    const double *ventThickness,
    int nVents,
    double PTvent
)
{
    int i, j;
    if (!computeIndex(i, j, rows, cols)) return;

    double h  = GET(Sh, cols, i, j);
    double Tn = GET(ST_next, cols, i, j);

    for (int k = 0; k < nVents; k++)
        if (i == ventY[k] && j == ventX[k])
        {
            h  += ventThickness[k];
            Tn = PTvent;
        }

    SET(Sh_next, cols, i, j, h);
    SET(ST_next, cols, i, j, Tn);
}

// ----------------------------------------------------------------------------
// computeOutflows
// ----------------------------------------------------------------------------
__global__ void computeOutflowsKernel(
    double *Sz,
    double *Sh,
    double *ST,
    double *Mf,
    int rows, int cols,
    double Pc,
    double a,
    double b,
    double c,
    double d
)
{
    const int Xi[MOORE_NEIGHBORS] = {0,-1,0,0,1,-1,1,1,-1};
    const int Xj[MOORE_NEIGHBORS] = {0,0,-1,1,0,-1,-1,1,1};

    int i, j;
    if (!computeIndex(i, j, rows, cols)) return;

    if (GET(Sh, cols, i, j) <= 0.0)
        return;

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];

    double T = GET(ST, cols, i, j);
    double rr = pow(10.0, a + b*T);
    double hc = pow(10.0, c + d*T);

    for (int k=0; k<MOORE_NEIGHBORS; k++)
    {
        double sz0 = GET(Sz, cols, i, j);
        double sz  = GET(Sz, cols, i+Xi[k], j+Xj[k]);

        h[k]  = GET(Sh, cols, i+Xi[k], j+Xj[k]);
        w[k]  = Pc;
        Pr[k] = rr;

        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz;
        else
            z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
    }

    H[0] = z[0];
    theta[0] = 0.0;
    eliminated[0] = false;

    for (int k=1; k<MOORE_NEIGHBORS; k++)
    {
        if (z[0] + h[0] > z[k] + h[k])
        {
            H[k] = z[k] + h[k];
            theta[k] = atan(( (z[0]+h[0]) - (z[k]+h[k]) ) / w[k]);
            eliminated[k] = false;
        }
        else 
            eliminated[k] = true;
    }

    bool loop;
    int counter;
    double avg;

    do {
        loop = false;
        avg = h[0];
        counter = 0;

        for (int k=0; k<MOORE_NEIGHBORS; k++)
            if (!eliminated[k]) { avg += H[k]; counter++; }

        if (counter != 0)
            avg /= (double)counter;

        for (int k=0; k<MOORE_NEIGHBORS; k++)
            if (!eliminated[k] && avg <= H[k])
                eliminated[k] = true, loop = true;

    } while(loop);

    for (int k=1; k<MOORE_NEIGHBORS; k++)
    {
        if (!eliminated[k] && h[0] > hc * cos(theta[k]))
            BUF_SET(Mf, rows, cols, k-1, i, j, Pr[k]*(avg-H[k]));
        else
            BUF_SET(Mf, rows, cols, k-1, i, j, 0.0);
    }
}

// ----------------------------------------------------------------------------
// massBalance
// ----------------------------------------------------------------------------
__global__ void massBalanceKernel(
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf,
    int rows, int cols
)
{
    const int Xi[MOORE_NEIGHBORS] = {0,-1,0,0,1,-1,1,1,-1};
    const int Xj[MOORE_NEIGHBORS] = {0,0,-1,1,0,-1,-1,1,1};
    const int inflows[NUMBER_OF_OUTFLOWS] = {3,2,1,0,6,7,4,5};

    int i, j;
    if (!computeIndex(i, j, rows, cols)) return;

    double initial_h = GET(Sh, cols, i, j);
    double initial_t = GET(ST, cols, i, j);

    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n=1; n<MOORE_NEIGHBORS; n++)
    {
        double neighT = GET(ST, cols, i+Xi[n], j+Xj[n]);
        double inF  = BUF_GET(Mf, rows, cols, inflows[n-1], i+Xi[n], j+Xj[n]);
        double outF = BUF_GET(Mf, rows, cols, n-1, i, j);

        h_next += inF - outF;
        t_next += (inF * neighT - outF * initial_t);
    }

    if (h_next > 0.0)
    {
        t_next /= h_next;
        SET(ST_next, cols, i, j, t_next);
        SET(Sh_next, cols, i, j, h_next);
    }
    else
    {
        SET(ST_next, cols, i, j, 0.0);
        SET(Sh_next, cols, i, j, 0.0);
    }
}

// ----------------------------------------------------------------------------
// computeNewTemperatureAndSolidification
// ----------------------------------------------------------------------------
__global__ void computeNewTempAndSolidKernel(
    double Pepsilon, double Psigma, double Pclock, double Pcool,
    double Prho, double Pcv, double Pac, double PTsol,
    double *Sz, double *Sz_next,
    double *Sh, double *Sh_next,
    double *ST, double *ST_next,
    double *Mhs, bool *Mb,
    int rows, int cols
)
{
    int i,j;
    if (!computeIndex(i, j, rows, cols)) return;

    double z = GET(Sz, cols, i, j);
    double h = GET(Sh, cols, i, j);
    double T = GET(ST, cols, i, j);
    bool   b = GET(Mb, cols, i, j);

    if (h > 0.0 && !b)
    {
        double aus =
            1.0 + (3.0 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool)
            / (Prho * Pcv * h * Pac);

        double nT = T / pow(aus, 1.0/3.0);

        if (nT > PTsol)
        {
            SET(ST_next, cols, i, j, nT);
            SET(Sh_next, cols, i, j, h);
            SET(Sz_next, cols, i, j, z);
        }
        else
        {
            SET(Sz_next, cols, i, j, z + h);
            SET(Sh_next, cols, i, j, 0.0);
            SET(ST_next, cols, i, j, PTsol);
            SET(Mhs,     cols, i, j, GET(Mhs, cols, i, j) + h);
        }
    }
    else
    {
        SET(Sz_next, cols, i, j, z);
        SET(Sh_next, cols, i, j, h);
        SET(ST_next, cols, i, j, T);
    }
}

// ----------------------------------------------------------------------------
// boundaryConditions (no-op come nel codice originale)
// ----------------------------------------------------------------------------
__global__ void boundaryConditionsKernel() {}

// ============================================================================
// REDUCTION — 2 PASSI (compatibile con GPU GTX 9xx)
// ============================================================================

// step 1: riduzione per blocco
__global__ void reduceBlocksKernel(const double *Sh, double *blockSums, int N)
{
    __shared__ double s[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    s[tid] = (idx < N ? Sh[idx] : 0.0);
    __syncthreads();

    for (int step=128; step>0; step>>=1)
    {
        if (tid < step)
            s[tid] += s[tid + step];
        __syncthreads();
    }

    if (tid == 0)
        blockSums[blockIdx.x] = s[0];
}

// ============================================================================
// WRAPPER FUNCTIONS (callable from sciara_fv2.cpp)
// ============================================================================

extern "C" {

// ----------------------------------------------------------------
// emitLava wrapper
// ----------------------------------------------------------------
void emitLavaCUDA(
    double *Sh, double *Sh_next, double *ST_next,
    int rows, int cols,
    int *ventX, int *ventY, double *ventThick, int nVents,
    double PTvent
)
{
    dim3 block(16,16);
    dim3 grid((cols+15)/16, (rows+15)/16);

    emitLavaKernel<<<grid,block>>>(
        Sh, Sh_next, ST_next,
        rows, cols,
        ventX, ventY, ventThick,
        nVents, PTvent
    );

    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------
// computeOutflows wrapper
// ----------------------------------------------------------------
void computeOutflowsCUDA(
    double *Sz, double *Sh, double *ST,
    double *Mf, int rows, int cols,
    double Pc, double a, double b, double c, double d
)
{
    dim3 block(16,16);
    dim3 grid((cols+15)/16, (rows+15)/16);

    computeOutflowsKernel<<<grid,block>>>(
        Sz, Sh, ST, Mf,
        rows, cols,
        Pc, a, b, c, d
    );

    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------
// massBalance wrapper
// ----------------------------------------------------------------
void massBalanceCUDA(
    double *Sh, double *Sh_next,
    double *ST, double *ST_next,
    double *Mf, int rows, int cols
)
{
    dim3 block(16,16);
    dim3 grid((cols+15)/16, (rows+15)/16);

    massBalanceKernel<<<grid,block>>>(
        Sh, Sh_next,
        ST, ST_next,
        Mf, rows, cols
    );

    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------
// computeNewTempAndSolidification wrapper
// ----------------------------------------------------------------
void computeNewTempAndSolidCUDA(
    double Pepsilon, double Psigma, double Pclock, double Pcool,
    double Prho, double Pcv, double Pac, double PTsol,
    double *Sz, double *Sz_next,
    double *Sh, double *Sh_next,
    double *ST, double *ST_next,
    double *Mhs, bool *Mb,
    int rows, int cols
)
{
    dim3 block(16,16);
    dim3 grid((cols+15)/16, (rows+15)/16);

    computeNewTempAndSolidKernel<<<grid,block>>>(
        Pepsilon, Psigma, Pclock, Pcool,
        Prho, Pcv, Pac, PTsol,
        Sz, Sz_next,
        Sh, Sh_next,
        ST, ST_next,
        Mhs, Mb,
        rows, cols
    );

    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------
// boundaryConditions wrapper
// ----------------------------------------------------------------
void boundaryConditionsCUDA() {}

// ----------------------------------------------------------------
// GLOBAL REDUCTION WRAPPER (2 step — safe for GTX 9xx)
// ----------------------------------------------------------------
double reduceAddCUDA(double *Sh, int rows, int cols)
{
    int N = rows * cols;

    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    double *blockSums;
    cudaMallocManaged(&blockSums, gridSize * sizeof(double));

    reduceBlocksKernel<<<gridSize, blockSize>>>(Sh, blockSums, N);
    cudaDeviceSynchronize();

    double total = 0.0;
    for (int i=0; i<gridSize; i++)
        total += blockSums[i];

    cudaFree(blockSums);
    return total;
}

} // extern "C"

