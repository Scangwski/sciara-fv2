#include "Sciara.h"
#include "io.h"
#include "util.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstring>

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define INPUT_PATH_ID          1
#define OUTPUT_PATH_ID         2
#define MAX_STEPS_ID           3
#define REDUCE_INTERVL_ID      4
#define THICKNESS_THRESHOLD_ID 5

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j)        ((M)[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) \
    ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) \
    ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// Host utility: simple CUDA error check
// ----------------------------------------------------------------------------
inline void cudaCheck(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// ----------------------------------------------------------------------------
// Host emitLava (rimane su CPU perché usa std::vector<TVent>)
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Device kernels (GLOBAL memory implementation)
// ----------------------------------------------------------------------------

__global__
void computeOutflows_kernel(
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
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // solo celle interne: i in [1, r-2], j in [1, c-2]
  if (i <= 0 || j <= 0 || i >= r-1 || j >= c-1)
    return;

  const int Xi[MOORE_NEIGHBORS] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
  const int Xj[MOORE_NEIGHBORS] = {0,  0, -1,  1,  0, -1, -1,  1,  1};

  bool   eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];
  double w[MOORE_NEIGHBORS];
  double Pr[MOORE_NEIGHBORS];
  // double f[MOORE_NEIGHBORS]; // non serve nella formula finale

  bool   loop;
  int    counter;
  double sz0, sz, T, avg, rr, hc;

  if (GET(Sh, c, i, j) <= 0.0)
    return;

  T  = GET(ST, c, i, j);
  rr = pow(10.0, _a + _b * T);
  hc = pow(10.0, _c + _d * T);

  // costruiamo z, h, w, Pr
  for (int k = 0; k < MOORE_NEIGHBORS; k++)
  {
    int ni = i + Xi[k];
    int nj = j + Xj[k];

    // con i,j interni e questo pattern, ni,nj sono SEMPRE in [0,r-1],[0,c-1]
    sz0   = GET(Sz, c, i,  j);
    sz    = GET(Sz, c, ni, nj);
    h[k]  = GET(Sh, c, ni, nj);
    w[k]  = Pc;
    Pr[k] = rr;

    if (k < VON_NEUMANN_NEIGHBORS)
      z[k] = sz;
    else
      z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
  }

  H[0]         = z[0];
  theta[0]     = 0.0;
  eliminated[0]= false;

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
  {
    if (z[0] + h[0] > z[k] + h[k])
    {
      H[k]         = z[k] + h[k];
      theta[k]     = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
      eliminated[k]= false;
    }
    else
    {
      eliminated[k]= true;
    }
  }

  // ciclo di eliminazione
  do
  {
    loop   = false;
    avg    = h[0];
    counter= 0;

    for (int k = 0; k < MOORE_NEIGHBORS; k++)
      if (!eliminated[k])
      {
        avg += H[k];
        counter++;
      }

    if (counter != 0)
      avg = avg / double(counter);

    for (int k = 0; k < MOORE_NEIGHBORS; k++)
      if (!eliminated[k] && avg <= H[k])
      {
        eliminated[k] = true;
        loop = true;
      }

  } while (loop);

  // calcolo dei flussi
  for (int k = 1; k < MOORE_NEIGHBORS; k++)
  {
    if (eliminated[k])
    {
      BUF_SET(Mf, r, c, k - 1, i, j, 0.0);
      continue;
    }

    if (h[0] > hc * cos(theta[k]))
      BUF_SET(Mf, r, c, k - 1, i, j, Pr[k] * (avg - H[k]));
    else
      BUF_SET(Mf, r, c, k - 1, i, j, 0.0);
  }
}

__global__
void massBalance_kernel(
    int r,
    int c,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i <= 0 || j <= 0 || i >= r-1 || j >= c-1)
    return;

  const int Xi[MOORE_NEIGHBORS] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
  const int Xj[MOORE_NEIGHBORS] = {0,  0, -1,  1,  0, -1, -1,  1,  1};
  const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

  double initial_h = GET(Sh, c, i, j);
  double initial_t = GET(ST, c, i, j);
  double h_next    = initial_h;
  double t_next    = initial_h * initial_t;

  for (int n = 1; n < MOORE_NEIGHBORS; n++)
  {
    int ni = i + Xi[n];
    int nj = j + Xj[n];

    // anche qui ni,nj sono in range con i,j interni
    double neigh_t = GET(ST, c, ni, nj);
    double inFlow  = BUF_GET(Mf, r, c, inflowsIndices[n - 1], ni, nj);
    double outFlow = BUF_GET(Mf, r, c, n - 1, i,  j);

    h_next += inFlow - outFlow;
    t_next += (inFlow * neigh_t - outFlow * initial_t);
  }

  if (h_next > 0.0)
  {
    t_next /= h_next;
    SET(ST_next, c, i, j, t_next);
    SET(Sh_next, c, i, j, h_next);
  }
  // se h_next <= 0, lasciamo Sh_next e ST_next come già impostati (da emitLava o step precedente)
}

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
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i <= 0 || j <= 0 || i >= r-1 || j >= c-1)
    return;

  double z = GET(Sz, c, i, j);
  double h = GET(Sh, c, i, j);
  double T = GET(ST, c, i, j);

  if (h > 0.0 && GET(Mb, c, i, j) == false)
  {
    double aus = 1.0 + (3.0 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) /
                        (Prho * Pcv * h * Pac);
    double nT  = T / pow(aus, 1.0 / 3.0);

    if (nT > PTsol) // no solidification
    {
      SET(ST_next, c, i, j, nT);
      // Sh_next viene già da massBalance (spessore aggiornato)
    }
    else // solidification
    {
      SET(Sz_next, c, i, j, z + h);
      SET(Sh_next, c, i, j, 0.0);
      SET(ST_next, c, i, j, PTsol);
      SET(Mhs,     c, i, j, GET(Mhs, c, i, j) + h);
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
  // Nel codice originale boundaryConditions è praticamente un NOP.
  // Manteniamo la stessa logica.
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i <= 0 || j <= 0 || i >= r-1 || j >= c-1)
    return;

  // Se un domani vuoi attivare vere BC:
  // if (GET(Mb, c, i, j))
  // {
  //   SET(Sh_next, c, i, j, 0.0);
  //   SET(ST_next, c, i, j, 0.0);
  // }
}

// ----------------------------------------------------------------------------
// Riduzione CUDA (GLOBAL) su Sh
// ----------------------------------------------------------------------------

__global__
void reduceAdd_kernel(double *in, double *partial, int N)
{
  extern __shared__ double sdata[];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < N)
    sdata[tid] = in[gid];
  else
    sdata[tid] = 0.0;

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    partial[blockIdx.x] = sdata[0];
}

double reduceAdd_gpu(int r, int c, double *buffer)
{
  int N       = r * c;
  int threads = 256;
  int blocks  = (N + threads - 1) / threads;

  double *partial = nullptr;
  cudaCheck(cudaMallocManaged(&partial, blocks * sizeof(double)),
            "cudaMallocManaged(partial)");

  reduceAdd_kernel<<<blocks, threads, threads * sizeof(double)>>>(buffer, partial, N);
  cudaCheck(cudaDeviceSynchronize(), "reduceAdd_kernel");

  double sum = 0.0;
  for (int i = 0; i < blocks; i++)
    sum += partial[i];

  cudaFree(partial);
  return sum;
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Sciara *sciara;
    init(sciara);

    // Input data
    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    int r = sciara->domain->rows;
    int c = sciara->domain->cols;

    dim3 blockDim(16,16);
    dim3 gridDim((c + blockDim.x - 1) / blockDim.x,
                 (r + blockDim.y - 1) / blockDim.y);

    // Simulation initialization
    simulationInitialize(sciara);

    util::Timer cl_timer;
    double total_current_lava = -1.0;

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

  

    // ---- MAIN LOOP ----
    while ((max_steps > 0 && sciara->simulation->step < max_steps) &&
       ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
        (total_current_lava == -1 || total_current_lava > thickness_threshold)))


    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;


        // Reset dei buffer next (GPU)
        cudaMemset(sciara->substates->Sh_next, 0, sizeof(double)*r*c);
        cudaMemset(sciara->substates->ST_next, 0, sizeof(double)*r*c);
        cudaMemset(sciara->substates->Sz_next, 0, sizeof(double)*r*c);

        // ---------------------------------------------
        // 🔥 EMIT LAVA (CPU) — versione originale!
        // ---------------------------------------------
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                emitLava(
                    i, j,
                    sciara->domain->rows,
                    sciara->domain->cols,
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
            }
        }

        // Copiamo Sh_next → Sh, ST_next → ST  (CPU)
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // ---------------------------------------------
        // computeOutflows (GPU)
        // ---------------------------------------------
        computeOutflows_kernel<<<gridDim, blockDim>>>(
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

        // ---------------------------------------------
        // massBalance (GPU)
        // ---------------------------------------------
        massBalance_kernel<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mf
        );

        // swap Sh/ST buffers
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // ---------------------------------------------
        // Temperature + Solidification (GPU)
        // ---------------------------------------------
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

        // swap again
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);
        std::swap(sciara->substates->Sz, sciara->substates->Sz_next);

        // boundary (GPU)
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

        // Reduction every N steps
        if (sciara->simulation->step % reduceInterval == 0)
        {
            cudaDeviceSynchronize();
            total_current_lava = reduceAdd_gpu(r, c, sciara->substates->Sh);
            printf("Step %d, current lava = %lf\n",
                sciara->simulation->step, total_current_lava);
        }
    }

    cudaDeviceSynchronize();

    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

    printf("Releasing memory...\n");
    finalize(sciara);
    return 0;
}
