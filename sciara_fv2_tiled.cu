#include "Sciara.h"
#include "io.h"
#include "util.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstring>

// ============================================================================
// I/O parameters
// ============================================================================
#define INPUT_PATH_ID          1
#define OUTPUT_PATH_ID         2
#define MAX_STEPS_ID           3
#define REDUCE_INTERVL_ID      4
#define THICKNESS_THRESHOLD_ID 5

// ============================================================================
// Buffer access macros
// ============================================================================
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j)        ((M)[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) \
    ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) \
    ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ============================================================================
// CUDA error check
// ============================================================================
inline void cudaCheck(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// ============================================================================
// emitLava (CPU) – unchanged
// ============================================================================
void emitLava(
    int i,
    int j,
    int r,
    int c,
    std::vector<TVent> &vent,
    double elapsed_time,
    double Pclock,
    double emission_time,
    double &total_emitted_lava,
    double Pac,
    double PTvent,
    double *Sh,
    double *Sh_next,
    double *ST_next)
{
  for (int k = 0; k < (int)vent.size(); k++)
    if (i == vent[k].y() && j == vent[k].x())
    {
      double thick = vent[k].thickness(elapsed_time, Pclock, emission_time, Pac);
      double cur_h = GET(Sh, c, i, j);
      SET(Sh_next, c, i, j, cur_h + thick);
      SET(ST_next, c, i, j, PTvent);
      total_emitted_lava += thick;
    }
}

// ============================================================================
//     TILED VERSION (NO HALO)
// ============================================================================
#define TILE_X 16
#define TILE_Y 16

__device__ __constant__ int dXi[MOORE_NEIGHBORS];
__device__ __constant__ int dXj[MOORE_NEIGHBORS];


// ============================================================================
// computeOutflows — TILED VERSION
// ============================================================================
template<int BX, int BY>
__global__
void computeOutflows_kernel_tiled(
    int r,
    int c,
    double *Sz,
    double *Sh,
    double *ST,
    double *Mf,
    double Pc,
    double _a,
    double _b,
    double _c,
    double _d)
{
    __shared__ double sSz[BY][BX];
    __shared__ double sSh[BY][BX];
    __shared__ double sST[BY][BX];

    int j = blockIdx.x * BX + threadIdx.x;
    int i = blockIdx.y * BY + threadIdx.y;

    if (i < r && j < c) {
        int idx = i * c + j;
        sSz[threadIdx.y][threadIdx.x] = Sz[idx];
        sSh[threadIdx.y][threadIdx.x] = Sh[idx];
        sST[threadIdx.y][threadIdx.x] = ST[idx];
    }

    __syncthreads();

    if (i <= 0 || j <= 0 || i >= r-1 || j >= c-1)
        return;

    // ---- shared memory helpers ----
    auto getShLocal = [&](int ni, int nj) {
        int li = threadIdx.y + (ni - i);
        int lj = threadIdx.x + (nj - j);
        if (li>=0 && li<BY && lj>=0 && lj<BX) return sSh[li][lj];
        return Sh[ni*c + nj];
    };

    auto getSzLocal = [&](int ni, int nj) {
        int li = threadIdx.y + (ni - i);
        int lj = threadIdx.x + (nj - j);
        if (li>=0 && li<BY && lj>=0 && lj<BX) return sSz[li][lj];
        return Sz[ni*c + nj];
    };

    auto getSTLocal = [&](int ni, int nj) {
        int li = threadIdx.y + (ni - i);
        int lj = threadIdx.x + (nj - j);
        if (li>=0 && li<BY && lj>=0 && lj<BX) return sST[li][lj];
        return ST[ni*c + nj];
    };

    // ---- logic identical to GLOBAL version ----
    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];

    if (getShLocal(i,j) <= 0.0) return;

    double T  = getSTLocal(i,j);
    double rr = pow(10.0, _a + _b*T);
    double hc = pow(10.0, _c + _d*T);
    double sz0 = getSzLocal(i,j);

    for (int k=0; k<MOORE_NEIGHBORS; k++) {
        int ni = i + dXi[k];
        int nj = j + dXj[k];
        double sz = getSzLocal(ni,nj);
        h[k] = getShLocal(ni,nj);
        w[k] = Pc;
        Pr[k]= rr;

        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz;
        else
            z[k] = sz0 - (sz0 - sz)/sqrt(2.0);
    }

    H[0] = z[0];
    theta[0] = 0.0;
    eliminated[0] = false;

    for (int k=1; k<MOORE_NEIGHBORS; k++) {
        if (z[0] + h[0] > z[k] + h[k]) {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0]+h[0])-(z[k]+h[k]))/w[k]);
            eliminated[k] = false;
        }
        else eliminated[k]=true;
    }

    bool loop;
    double avg;
    int counter;

    do {
        loop = false;
        avg = h[0];
        counter = 0;

        for (int k=0; k<MOORE_NEIGHBORS; k++)
            if (!eliminated[k]) { avg += H[k]; counter++; }

        if (counter>0) avg /= double(counter);

        for (int k=0; k<MOORE_NEIGHBORS; k++)
            if (!eliminated[k] && avg <= H[k]) {
                eliminated[k]=true;
                loop=true;
            }

    } while(loop);

    int idx = i*c + j;

    for (int k=1; k<MOORE_NEIGHBORS; k++) {
        if (!eliminated[k] && h[0] > hc*cos(theta[k]))
            BUF_SET(Mf,r,c,k-1,i,j, Pr[k]*(avg-H[k]));
        else
            BUF_SET(Mf,r,c,k-1,i,j, 0.0);
    }
}


// ============================================================================
// massBalance — TILED VERSION
// ============================================================================
template<int BX, int BY>
__global__
void massBalance_kernel_tiled(
    int r,
    int c,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf)
{
    __shared__ double sSh[BY][BX];
    __shared__ double sST[BY][BX];

    int j = blockIdx.x * BX + threadIdx.x;
    int i = blockIdx.y * BY + threadIdx.y;

    if (i < r && j < c) {
        int idx = i*c + j;
        sSh[threadIdx.y][threadIdx.x] = Sh[idx];
        sST[threadIdx.y][threadIdx.x] = ST[idx];
    }
    __syncthreads();

    if (i <= 0 || j <= 0 || i >= r-1 || j >= c-1)
        return;

    auto getShLocal = [&](int ni,int nj){
        int li = threadIdx.y + (ni-i);
        int lj = threadIdx.x + (nj-j);
        if (li>=0 && li<BY && lj>=0 && lj<BX) return sSh[li][lj];
        return Sh[ni*c + nj];
    };

    auto getSTLocal = [&](int ni,int nj){
        int li = threadIdx.y + (ni-i);
        int lj = threadIdx.x + (nj-j);
        if (li>=0 && li<BY && lj>=0 && lj<BX) return sST[li][lj];
        return ST[ni*c + nj];
    };

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3,2,1,0,6,7,4,5};

    double h0 = getShLocal(i,j);
    double t0 = getSTLocal(i,j);

    double h_new = h0;
    double t_new = h0*t0;

    for (int n=1; n<MOORE_NEIGHBORS; n++){
        int ni = i + dXi[n];
        int nj = j + dXj[n];

        double neighT = getSTLocal(ni,nj);
        double inFlow = BUF_GET(Mf,r,c,inflowsIndices[n-1],ni,nj);
        double outFlow= BUF_GET(Mf,r,c,n-1,i,j);

        h_new += inFlow - outFlow;
        t_new += inFlow*neighT - outFlow*t0;
    }

    if (h_new > 0){
        t_new /= h_new;
        SET(Sh_next,c,i,j,h_new);
        SET(ST_next,c,i,j,t_new);
    }
}


// ============================================================================
// Other kernels unchanged
// ============================================================================
__global__
void computeNewTemperatureAndSolidification_kernel(
    int r,
    int c,
    double Pepsilon,
    double Psigma,
    double Pclock,
    double Pcool,
    double Prho,
    double Pcv,
    double Pac,
    double PTsol,
    double *Sz,
    double *Sz_next,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf,
    double *Mhs,
    bool   *Mb)
{
    // unchanged
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;

    if(i<=0||j<=0||i>=r-1||j>=c-1) return;

    double z=GET(Sz,c,i,j);
    double h=GET(Sh,c,i,j);
    double T=GET(ST,c,i,j);

    if(h>0.0 && GET(Mb,c,i,j)==false){
        double aus = 1.0 + (3.0 * pow(T,3.0)*Pepsilon*Psigma*Pclock*Pcool)/(Prho*Pcv*h*Pac);
        double nT = T / pow(aus,1.0/3.0);

        if(nT > PTsol){
            SET(ST_next,c,i,j,nT);
        }
        else{
            SET(Sz_next,c,i,j,z+h);
            SET(Sh_next,c,i,j,0.0);
            SET(ST_next,c,i,j,PTsol);
            SET(Mhs,c,i,j, GET(Mhs,c,i,j)+h);
        }
    }
}

__global__
void boundaryConditions_kernel(
    int r,
    int c,
    double *Mf,
    bool   *Mb,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next)
{
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<=0||j<=0||i>=r-1||j>=c-1) return;
}


// ============================================================================
// Reduction unchanged
// ============================================================================
__global__
void reduceAdd_kernel(double *in, double *partial, int N)
{
    extern __shared__ double sdata[];
    int tid=threadIdx.x;
    int gid=blockIdx.x*blockDim.x + tid;

    sdata[tid]=(gid<N? in[gid]: 0.0);
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if(tid<s) sdata[tid]+=sdata[tid+s];
        __syncthreads();
    }
    if(tid==0) partial[blockIdx.x]=sdata[0];
}

double reduceAdd_gpu(int r,int c,double *buffer)
{
    int N=r*c;
    int threads=256;
    int blocks=(N+threads-1)/threads;

    double *partial;
    cudaMallocManaged(&partial, blocks*sizeof(double));
    reduceAdd_kernel<<<blocks,threads,threads*sizeof(double)>>>(buffer,partial,N);
    cudaDeviceSynchronize();

    double s=0;
    for(int i=0;i<blocks;i++) s+=partial[i];
    cudaFree(partial);
    return s;
}


// ============================================================================
// MAIN FUNCTION — updated only where necessary
// ============================================================================
int main(int argc, char **argv)
{
    Sciara *sciara;
    init(sciara);

    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    int r = sciara->domain->rows;
    int c = sciara->domain->cols;

    // Upload neighbor arrays
    int Xi_host[MOORE_NEIGHBORS]={0,-1,0,0,1,-1,1,1,-1};
    int Xj_host[MOORE_NEIGHBORS]={0,0,-1,1,0,-1,-1,1,1};
    cudaMemcpyToSymbol(dXi, Xi_host, sizeof(int)*MOORE_NEIGHBORS);
    cudaMemcpyToSymbol(dXj, Xj_host, sizeof(int)*MOORE_NEIGHBORS);

    dim3 blockDim(TILE_X, TILE_Y);
    dim3 gridDim((c+TILE_X-1)/TILE_X,(r+TILE_Y-1)/TILE_Y);

    simulationInitialize(sciara);

    util::Timer cl_timer;
    double total_current_lava = -1.0;

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    // ========================================================================
    // MAIN LOOP
    // ========================================================================
    while ((max_steps > 0 && sciara->simulation->step < max_steps) &&
       ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
        (total_current_lava == -1 || total_current_lava > thickness_threshold)))

    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        cudaMemset(sciara->substates->Sh_next,0,sizeof(double)*r*c);
        cudaMemset(sciara->substates->ST_next,0,sizeof(double)*r*c);
        cudaMemset(sciara->substates->Sz_next,0,sizeof(double)*r*c);

        // ---------------- EMIT LAVA (CPU) ----------------
        for(int i=0;i<r;i++)
            for(int j=0;j<c;j++)
                emitLava(i,j,
                    r,c,
                    sciara->simulation->vent,
                    sciara->simulation->elapsed_time,
                    sciara->parameters->Pclock,
                    sciara->simulation->emission_time,
                    sciara->simulation->total_emitted_lava,
                    sciara->parameters->Pac,
                    sciara->parameters->PTvent,
                    sciara->substates->Sh,
                    sciara->substates->Sh_next,
                    sciara->substates->ST_next
                );

        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // ---------------- OUTFLOWS (TILED) ----------------
        computeOutflows_kernel_tiled<TILE_X,TILE_Y><<<gridDim,blockDim>>>(
            r,c,
            sciara->substates->Sz,
            sciara->substates->Sh,
            sciara->substates->ST,
            sciara->substates->Mf,
            sciara->parameters->Pc,
            sciara->parameters->a,
            sciara->parameters->b,
            sciara->parameters->c,
            sciara->parameters->d
        );

        // ---------------- MASS BALANCE (TILED) ----------------
        massBalance_kernel_tiled<TILE_X,TILE_Y><<<gridDim,blockDim>>>(
            r,c,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mf
        );

        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // ---------------- TEMPERATURE + SOLIDIFICATION ----------------
        computeNewTemperatureAndSolidification_kernel<<<gridDim,blockDim>>>(
            r,c,
            sciara->parameters->Pepsilon,
            sciara->parameters->Psigma,
            sciara->parameters->Pclock,
            sciara->parameters->Pcool,
            sciara->parameters->Prho,
            sciara->parameters->Pcv,
            sciara->parameters->Pac,
            sciara->parameters->PTsol,
            sciara->substates->Sz,
            sciara->substates->Sz_next,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mf,
            sciara->substates->Mhs,
            sciara->substates->Mb
        );

        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);
        std::swap(sciara->substates->Sz, sciara->substates->Sz_next);

        // ---------------- BOUNDARY CONDITIONS ----------------
        boundaryConditions_kernel<<<gridDim,blockDim>>>(
            r,c,
            sciara->substates->Mf,
            sciara->substates->Mb,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next
        );

        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // ---------------- REDUCTION ----------------
        if (sciara->simulation->step % reduceInterval == 0)
        {
            cudaDeviceSynchronize();
            total_current_lava = reduceAdd_gpu(r,c,sciara->substates->Sh);
            printf("Step %d, current lava = %lf\n",
                sciara->simulation->step, total_current_lava);
        }
    }

    cudaDeviceSynchronize();

    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);
    finalize(sciara);

    return 0;
}
