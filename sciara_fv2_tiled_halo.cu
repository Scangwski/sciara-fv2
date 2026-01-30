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
// computeOutflows — TILED WITH HALO
// ============================================================================
template<int BX, int BY>
__global__
void computeOutflows_kernel_tiled_wHalo(
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
    // Shared memory (+2 halo cells)
    __shared__ double sh_Sz[BY+2][BX+2];
    __shared__ double sh_Sh[BY+2][BX+2];
    __shared__ double sh_ST[BY+2][BX+2];

    int gj = blockIdx.x * BX + threadIdx.x;   // global j
    int gi = blockIdx.y * BY + threadIdx.y;   // global i

    int lj = threadIdx.x + 1; // local j index inside tile + halo
    int li = threadIdx.y + 1; // local i index inside tile + halo

    // ---------------------- LOAD CENTER CELL ----------------------
    if (gi < r && gj < c) {
        int idx = gi * c + gj;
        sh_Sz[li][lj] = Sz[idx];
        sh_Sh[li][lj] = Sh[idx];
        sh_ST[li][lj] = ST[idx];
    }

    // ---------------------- LOAD HALO -----------------------------
    // Left halo
    if (threadIdx.x == 0 && gj > 0 && gi < r) {
        sh_Sz[li][0]     = Sz[gi*c + (gj-1)];
        sh_Sh[li][0]     = Sh[gi*c + (gj-1)];
        sh_ST[li][0]     = ST[gi*c + (gj-1)];
    }
    // Right halo
    if (threadIdx.x == BX-1 && gj < c-1 && gi < r) {
        sh_Sz[li][BX+1] = Sz[gi*c + (gj+1)];
        sh_Sh[li][BX+1] = Sh[gi*c + (gj+1)];
        sh_ST[li][BX+1] = ST[gi*c + (gj+1)];
    }
    // Top halo
    if (threadIdx.y == 0 && gi > 0 && gj < c) {
        sh_Sz[0][lj]     = Sz[(gi-1)*c + gj];
        sh_Sh[0][lj]     = Sh[(gi-1)*c + gj];
        sh_ST[0][lj]     = ST[(gi-1)*c + gj];
    }
    // Bottom halo
    if (threadIdx.y == BY-1 && gi < r-1 && gj < c) {
        sh_Sz[BY+1][lj] = Sz[(gi+1)*c + gj];
        sh_Sh[BY+1][lj] = Sh[(gi+1)*c + gj];
        sh_ST[BY+1][lj] = ST[(gi+1)*c + gj];
    }

    // ---------------------- CORNER HALO --------------------------
    if (threadIdx.x == 0 && threadIdx.y == 0 && gi>0 && gj>0) {
        sh_Sz[0][0] = Sz[(gi-1)*c + (gj-1)];
        sh_Sh[0][0] = Sh[(gi-1)*c + (gj-1)];
        sh_ST[0][0] = ST[(gi-1)*c + (gj-1)];
    }
    if (threadIdx.x == BX-1 && threadIdx.y == 0 && gi>0 && gj<c-1) {
        sh_Sz[0][BX+1] = Sz[(gi-1)*c + (gj+1)];
        sh_Sh[0][BX+1] = Sh[(gi-1)*c + (gj+1)];
        sh_ST[0][BX+1] = ST[(gi-1)*c + (gj+1)];
    }
    if (threadIdx.x == 0 && threadIdx.y == BY-1 && gi<r-1 && gj>0) {
        sh_Sz[BY+1][0] = Sz[(gi+1)*c + (gj-1)];
        sh_Sh[BY+1][0] = Sh[(gi+1)*c + (gj-1)];
        sh_ST[BY+1][0] = ST[(gi+1)*c + (gj-1)];
    }
    if (threadIdx.x == BX-1 && threadIdx.y == BY-1 && gi<r-1 && gj<c-1) {
        sh_Sz[BY+1][BX+1] = Sz[(gi+1)*c + (gj+1)];
        sh_Sh[BY+1][BX+1] = Sh[(gi+1)*c + (gj+1)];
        sh_ST[BY+1][BX+1] = ST[(gi+1)*c + (gj+1)];
    }

    __syncthreads();

    // Avoid boundary
    if (gi <= 0 || gj <= 0 || gi >= r-1 || gj >= c-1)
        return;

    auto getSzL = [&](int k) { return sh_Sz[li + dXi[k]][lj + dXj[k]]; };
    auto getShL = [&](int k) { return sh_Sh[li + dXi[k]][lj + dXj[k]]; };
    auto getSTL = [&](int k) { return sh_ST[li + dXi[k]][lj + dXj[k]]; };

    // ---------------- MODEL LOGIC (identical to serial) --------------------

    double h0 = sh_Sh[li][lj];
    if (h0 <= 0.0) return;

    double T = sh_ST[li][lj];
    double rr = pow(10.0, _a + _b*T);
    double hc = pow(10.0, _c + _d*T);

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS], h[MOORE_NEIGHBORS], H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double sz0 = sh_Sz[li][lj];

    // Load neighbors in shared memory
    for (int k=0;k<MOORE_NEIGHBORS;k++){
        z[k] = getSzL(k);
        if (k >= VON_NEUMANN_NEIGHBORS)
            z[k] = sz0 - (sz0 - z[k]) / sqrt(2.0);

        h[k] = getShL(k);
        theta[k] = 0.0;
        eliminated[k] = false;
    }

    H[0] = z[0];

    for (int k=1;k<MOORE_NEIGHBORS;k++){
        if (z[0] + h0 > z[k] + h[k]) {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0] + h0) - (z[k] + h[k])) / Pc);
        } 
        else eliminated[k] = true;
    }

    bool loop;
    double avg = 0.0;   // <--------------- FIX: declare here

    do {
        loop = false;
        avg = h0;
        int count = 0;

        for (int k = 0; k < MOORE_NEIGHBORS; k++)
            if (!eliminated[k]) {
                avg += H[k];
                count++;
            }

        avg /= count;

        for (int k = 0; k < MOORE_NEIGHBORS; k++)
            if (!eliminated[k] && avg <= H[k]) {
                eliminated[k] = true;
                loop = true;
            }

    } while(loop);


    // ---------------- WRITE OUTFLOWS ----------------
    int base = gi * c + gj;

    for (int k = 1; k < MOORE_NEIGHBORS; k++)
    {
        if (!eliminated[k] && h0 > hc * cos(theta[k])) {
            BUF_SET(Mf, r, c, k - 1, gi, gj, rr * (avg - H[k]));
        } else {
            BUF_SET(Mf, r, c, k - 1, gi, gj, 0.0);
        }
    }
}

// ============================================================================
// massBalance — TILED WITH HALO
// ============================================================================
template<int BX, int BY>
__global__
void massBalance_kernel_tiled_wHalo(
    int r,
    int c,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf)
{
    __shared__ double sh_Sh[BY+2][BX+2];
    __shared__ double sh_ST[BY+2][BX+2];

    int gj = blockIdx.x*BX + threadIdx.x;
    int gi = blockIdx.y*BY + threadIdx.y;

    int lj = threadIdx.x + 1;
    int li = threadIdx.y + 1;

    if (gi < r && gj < c){
        int idx = gi*c + gj;
        sh_Sh[li][lj] = Sh[idx];
        sh_ST[li][lj] = ST[idx];
    }

    // HALO LEFT
    if (threadIdx.x==0 && gj>0 && gi<r){
        sh_Sh[li][0] = Sh[gi*c + (gj-1)];
        sh_ST[li][0] = ST[gi*c + (gj-1)];
    }
    // HALO RIGHT
    if (threadIdx.x==BX-1 && gj<c-1 && gi<r){
        sh_Sh[li][BX+1] = Sh[gi*c + (gj+1)];
        sh_ST[li][BX+1] = ST[gi*c + (gj+1)];
    }
    // HALO TOP
    if (threadIdx.y==0 && gi>0 && gj<c){
        sh_Sh[0][lj] = Sh[(gi-1)*c + gj];
        sh_ST[0][lj] = ST[(gi-1)*c + gj];
    }
    // HALO BOTTOM
    if (threadIdx.y==BY-1 && gi<r-1 && gj<c){
        sh_Sh[BY+1][lj] = Sh[(gi+1)*c + gj];
        sh_ST[BY+1][lj] = ST[(gi+1)*c + gj];
    }

    __syncthreads();

    if (gi<=0 || gj<=0 || gi>=r-1 || gj>=c-1)
        return;

    auto ShL = [&](int k){ return sh_Sh[li + dXi[k]][lj + dXj[k]]; };
    auto STL = [&](int k){ return sh_ST[li + dXi[k]][lj + dXj[k]]; };

    const int inflows[8]={3,2,1,0,6,7,4,5};

    double h = ShL(0);
    double T = STL(0);

    double h_new = h;
    double t_new = h*T;

    for(int n=1;n<9;n++){
        int ni = gi + dXi[n];
        int nj = gj + dXj[n];
        double neighT = STL(n);
        double inF  = BUF_GET(Mf,r,c,inflows[n-1],ni,nj);
        double ouF  = BUF_GET(Mf,r,c,n-1,gi,gj);
        h_new += inF - ouF;
        t_new += inF*neighT - ouF*T;
    }

    if (h_new > 0){
        t_new /= h_new;
        SET(Sh_next,c,gi,gj,h_new);
        SET(ST_next,c,gi,gj,t_new);
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
    printf("RUNNING %s | build %s %s\n", __FILE__, __DATE__, __TIME__);

    Sciara *sciara;
    init(sciara);

    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    int r = sciara->domain->rows;
    int c = sciara->domain->cols;

    // Upload neighbor arrays
    int Xi_host[MOORE_NEIGHBORS] = {0,-1,0,0,1,-1,1,1,-1};
    int Xj_host[MOORE_NEIGHBORS] = {0,0,-1,1,0,-1,-1,1,1};
    cudaMemcpyToSymbol(dXi, Xi_host, sizeof(int)*MOORE_NEIGHBORS);
    cudaMemcpyToSymbol(dXj, Xj_host, sizeof(int)*MOORE_NEIGHBORS);

    dim3 blockDim(TILE_X, TILE_Y);
    dim3 gridDim((c + TILE_X - 1) / TILE_X, (r + TILE_Y - 1) / TILE_Y);

    simulationInitialize(sciara);

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    double total_current_lava = -1.0;

    // IMPORTANT: start timing after init is done
    cudaDeviceSynchronize();
    util::Timer cl_timer;

    // ========================================================================
    // MAIN LOOP
    // ========================================================================
    while ((max_steps > 0 && sciara->simulation->step < max_steps) &&
           ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
            (total_current_lava == -1 || total_current_lava > thickness_threshold)))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        cudaMemset(sciara->substates->Sh_next, 0, sizeof(double)*r*c);
        cudaMemset(sciara->substates->ST_next, 0, sizeof(double)*r*c);
        cudaMemset(sciara->substates->Sz_next, 0, sizeof(double)*r*c);

        // ---------------- EMIT LAVA (CPU) ----------------
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                emitLava(i, j,
                    r, c,
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
        computeOutflows_kernel_tiled_wHalo<TILE_X, TILE_Y><<<gridDim, blockDim>>>(
            r, c,
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
        massBalance_kernel_tiled_wHalo<TILE_X, TILE_Y><<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mf
        );

        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // ---------------- TEMPERATURE + SOLIDIFICATION ----------------
        computeNewTemperatureAndSolidification_kernel<<<gridDim, blockDim>>>(
            r, c,
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
        boundaryConditions_kernel<<<gridDim, blockDim>>>(
            r, c,
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
        if (reduceInterval > 0 && (sciara->simulation->step % reduceInterval == 0))
        {
            cudaDeviceSynchronize();
            total_current_lava = reduceAdd_gpu(r, c, sciara->substates->Sh);
        }
    }

    cudaDeviceSynchronize();

    // Final prints (same style as colleague)
    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Final Step %d\n", sciara->simulation->step);
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}
