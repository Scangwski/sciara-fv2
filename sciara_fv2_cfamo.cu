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

#define TILE_X 16
#define TILE_Y 16

__device__ __constant__ int dXi[MOORE_NEIGHBORS];
__device__ __constant__ int dXj[MOORE_NEIGHBORS];
__device__ __constant__ int dOpp[MOORE_NEIGHBORS];



template<int BX, int BY>
__global__
void cfamo_fused_kernel_tiled_halo2(
    int r, int c,
    double *Sz,
    double *Sh,
    double *ST,
    double *Sh_next,
    double *ST_next,
    double Pc,
    double _a, double _b, double _c, double _d)
{
    // halo radius=2
    __shared__ double sh_Sz[BY+4][BX+4];
    __shared__ double sh_Sh[BY+4][BX+4];
    __shared__ double sh_ST[BY+4][BX+4];

    // OUTFLOWS SOLO PER TILE (no halo1)
    __shared__ double sh_F[MOORE_NEIGHBORS][BY][BX]; // k=1..8 usati, k=0 ignorato

    const int gj = blockIdx.x * BX + threadIdx.x;
    const int gi = blockIdx.y * BY + threadIdx.y;

    const int tx = threadIdx.x;     // 0..BX-1
    const int ty = threadIdx.y;     // 0..BY-1
    const int lj2 = tx + 2;         // halo2 offset
    const int li2 = ty + 2;

    // ----------------------------
    // 1) LOAD halo2 in shared (uguale al tuo cfame)
    // ----------------------------
    for (int dy = threadIdx.y; dy < BY+4; dy += BY) {
        int giy = blockIdx.y * BY + (dy - 2);
        for (int dx = threadIdx.x; dx < BX+4; dx += BX) {
            int gjx = blockIdx.x * BX + (dx - 2);

            double z=0.0, h=0.0, t=0.0;
            if (giy >= 0 && giy < r && gjx >= 0 && gjx < c) {
                int idx = giy * c + gjx;
                z = Sz[idx]; h = Sh[idx]; t = ST[idx];
            }
            sh_Sz[dy][dx]=z;
            sh_Sh[dy][dx]=h;
            sh_ST[dy][dx]=t;
        }
    }
    __syncthreads();

    // ----------------------------
    // helper: calcola TUTTI gli outflow di una cella in halo2 (li,lj) e li mette in out[1..8]
    // con gestione bordi globali (come hai fatto in CfAMe)
    // ----------------------------
    auto compute_outflows_all = [&](int gi_loc, int gj_loc, int li, int lj, double out[MOORE_NEIGHBORS])
    {
        // default zero
        #pragma unroll
        for (int k=0;k<MOORE_NEIGHBORS;k++) out[k]=0.0;

        // bordo globale -> zero
        if (gi_loc <= 0 || gj_loc <= 0 || gi_loc >= r-1 || gj_loc >= c-1) return;

        double h0 = sh_Sh[li][lj];
        if (h0 <= 0.0) return;

        double T  = sh_ST[li][lj];
        double rr = pow(10.0, _a + _b*T);
        double hc = pow(10.0, _c + _d*T);

        bool eliminated[MOORE_NEIGHBORS];
        double z[MOORE_NEIGHBORS], h[MOORE_NEIGHBORS], H[MOORE_NEIGHBORS];
        double theta[MOORE_NEIGHBORS];

        double sz0 = sh_Sz[li][lj];

        #pragma unroll
        for (int k=0;k<MOORE_NEIGHBORS;k++){
            z[k] = sh_Sz[li + dXi[k]][lj + dXj[k]];
            if (k >= VON_NEUMANN_NEIGHBORS)
                z[k] = sz0 - (sz0 - z[k]) / sqrt(2.0);
            h[k] = sh_Sh[li + dXi[k]][lj + dXj[k]];
            theta[k]=0.0;
            eliminated[k]=false;
        }

        H[0] = z[0];
        for (int k=1;k<MOORE_NEIGHBORS;k++){
            if (z[0] + h0 > z[k] + h[k]) {
                H[k] = z[k] + h[k];
                theta[k] = atan(((z[0]+h0) - (z[k]+h[k])) / Pc);
            } else {
                eliminated[k]=true;
            }
        }

        bool loop;
        double avg=0.0;
        do{
            loop=false;
            avg=h0;
            int count=0;
            for(int k=0;k<MOORE_NEIGHBORS;k++)
                if(!eliminated[k]){ avg += H[k]; count++; }
            avg /= count;

            for(int k=0;k<MOORE_NEIGHBORS;k++)
                if(!eliminated[k] && avg <= H[k]){
                    eliminated[k]=true;
                    loop=true;
                }
        }while(loop);

        for(int k=1;k<MOORE_NEIGHBORS;k++){
            if(!eliminated[k] && h0 > hc*cos(theta[k])){
                out[k] = rr * (avg - H[k]);
            }
        }
    };

    // ----------------------------
    // 2) COMPUTE OUTFLOWS per la cella corrente (solo tile interno)
    // ----------------------------
    double myF[MOORE_NEIGHBORS];
    compute_outflows_all(gi, gj, li2, lj2, myF);

    // salva in shared SOLO per tile interno
    #pragma unroll
    for (int k=1;k<MOORE_NEIGHBORS;k++){
        if (gi < r && gj < c) sh_F[k][ty][tx] = myF[k];
        else sh_F[k][ty][tx] = 0.0;
    }

    __syncthreads();

    // ----------------------------
    // 3) MASS BALANCE in GATHER (conflict-free)
    // ----------------------------
    // come nel codice originale: non aggiorniamo i bordi globali
    if (gi <= 0 || gj <= 0 || gi >= r-1 || gj >= c-1) return;

    double h = sh_Sh[li2][lj2];
    double T = sh_ST[li2][lj2];

    double h_new = h;
    double t_new = h * T;

    #pragma unroll
    for(int n=1;n<MOORE_NEIGHBORS;n++){
        // outflow mio verso n
        double ouF = sh_F[n][ty][tx];

        // vicino globale e locale
        int ngi = gi + dXi[n];
        int ngj = gj + dXj[n];
        int nty = ty + dXi[n];
        int ntx = tx + dXj[n];

        // temperatura del vicino (da halo2)
        double neighT = sh_ST[li2 + dXi[n]][lj2 + dXj[n]];

        // inflow: outflow del vicino nella direzione opposta
        double inF = 0.0;

        // Se il vicino è dentro tile => leggi shared
        if (ntx >= 0 && ntx < BX && nty >= 0 && nty < BY) {
            inF = sh_F[dOpp[n]][nty][ntx];
        } else {
            // vicino fuori tile => ricalcolo on-the-fly (CfAMo)
            double neighF[MOORE_NEIGHBORS];
            // nota: il vicino in halo2 si trova a (li2+dXi[n], lj2+dXj[n])
            compute_outflows_all(ngi, ngj, li2 + dXi[n], lj2 + dXj[n], neighF);
            inF = neighF[dOpp[n]];
        }

        h_new += inF - ouF;
        t_new += inF * neighT - ouF * T;
    }

    if (h_new > 0.0){
        t_new /= h_new;
        Sh_next[gi*c + gj] = h_new;
        ST_next[gi*c + gj] = t_new;
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

    int hOpp[MOORE_NEIGHBORS];

    // centro: non usato
    hOpp[0] = 0;

    for (int k = 1; k < MOORE_NEIGHBORS; ++k) {
        hOpp[k] = -1;
        for (int q = 1; q < MOORE_NEIGHBORS; ++q) {
            if (Xi_host[q] == -Xi_host[k] &&
                Xj_host[q] == -Xj_host[k]) {
                hOpp[k] = q;
                break;
            }
        }
    }

    cudaMemcpyToSymbol(dOpp, hOpp, sizeof(int)*MOORE_NEIGHBORS);


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

        // ---------------- CfAMe (FUSED OUTFLOWS + MASS BALANCE) ----------------
cfamo_fused_kernel_tiled_halo2<TILE_X, TILE_Y><<<gridDim, blockDim>>>(
    r, c,
    sciara->substates->Sz,
    sciara->substates->Sh,
    sciara->substates->ST,
    sciara->substates->Sh_next,
    sciara->substates->ST_next,
    sciara->parameters->Pc,
    sciara->parameters->a,
    sciara->parameters->b,
    sciara->parameters->c,
    sciara->parameters->d
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
