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
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// Host utility: simple CUDA error check (optional but useful)
// ----------------------------------------------------------------------------
inline void cudaCheck(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// ----------------------------------------------------------------------------
// Host emitLava (lasciata su CPU: usa std::vector<TVent>)
// ----------------------------------------------------------------------------
void emitLava(
    int i,
    int j,
    int r,
    int c,
    vector<TVent> &vent,
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
      SET(Sh_next, c, i, j, GET(Sh, c, i, j) + thick);
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
    int *Xi,
    int *Xj,
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

if (i <= 0 || j <= 0 || i >= r-1 || j >= c-1)
    return;


  bool eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];
  double w[MOORE_NEIGHBORS];  // Distances between central and adjacent cells
  double Pr[MOORE_NEIGHBORS]; // Relaxation rate array
  double f[MOORE_NEIGHBORS];
  bool loop;
  int counter;
  double sz0, sz, T, avg, rr, hc;

  if (GET(Sh, c, i, j) <= 0)
    return;

  T  = GET(ST, c, i, j);
  rr = pow(10.0, _a + _b * T);
  hc = pow(10.0, _c + _d * T);

  for (int k = 0; k < MOORE_NEIGHBORS; k++)
  {
    sz0 = GET(Sz, c, i, j);
    sz  = GET(Sz, c, i + Xi[k], j + Xj[k]);
    h[k] = GET(Sh, c, i + Xi[k], j + Xj[k]);
    w[k] = Pc;
    Pr[k] = rr;

    if (k < VON_NEUMANN_NEIGHBORS)
      z[k] = sz;
    else
      z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
  }

  H[0]       = z[0];
  theta[0]   = 0.0;
  eliminated[0] = false;

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
    if (z[0] + h[0] > z[k] + h[k])
    {
      H[k]        = z[k] + h[k];
      theta[k]    = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
      eliminated[k] = false;
    }
    else
    {
      eliminated[k] = true;
    }

  do
  {
    loop = false;
    avg  = h[0];
    counter = 0;
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

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
    if (!eliminated[k] && h[0] > hc * cos(theta[k]))
      BUF_SET(Mf, r, c, k - 1, i, j, Pr[k] * (avg - H[k]));
    else
      BUF_SET(Mf, r, c, k - 1, i, j, 0.0);
}

__global__
void massBalance_kernel(
    int r,
    int c,
    int *Xi,
    int *Xj,
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


  const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};
  double inFlow;
  double outFlow;
  double neigh_t;
  double initial_h = GET(Sh, c, i, j);
  double initial_t = GET(ST, c, i, j);
  double h_next    = initial_h;
  double t_next    = initial_h * initial_t;

  for (int n = 1; n < MOORE_NEIGHBORS; n++)
  {
    neigh_t = GET(ST, c, i + Xi[n], j + Xj[n]);
    inFlow  = BUF_GET(Mf, r, c, inflowsIndices[n - 1], i + Xi[n], j + Xj[n]);

    outFlow = BUF_GET(Mf, r, c, n - 1, i, j);

    h_next += inFlow - outFlow;
    t_next += (inFlow * neigh_t - outFlow * initial_t);
  }

  if (h_next > 0.0)
  {
    t_next /= h_next;
    SET(ST_next, c, i, j, t_next);
    SET(Sh_next, c, i, j, h_next);
  }
  // else: lascia Sh_next e ST_next come erano (aggiornate altrove / inizializzate)
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


  double nT, aus;
  double z = GET(Sz, c, i, j);
  double h = GET(Sh, c, i, j);
  double T = GET(ST, c, i, j);

  if (h > 0.0 && GET(Mb, c, i, j) == false)
  {
    aus = 1.0 + (3.0 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
    nT  = T / pow(aus, 1.0 / 3.0);

    if (nT > PTsol) // no solidification
      SET(ST_next, c, i, j, nT);
    else // solidification
    {
      SET(Sz_next, c, i, j, z + h);
      SET(Sh_next, c, i, j, 0.0);
      SET(ST_next, c, i, j, PTsol);
      SET(Mhs, c, i, j, GET(Mhs, c, i, j) + h);
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
  // Nota: nel codice originale boundaryConditions è vuota (return immediato),
  // quindi qui manteniamo lo stesso comportamento.
  // Se in futuro vorrai applicare condizioni al contorno reali, puoi farlo qui.

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= 0 || j <= 0 || i >= r-1 || j >= c-1)
    return;


  // Se vuoi attivare BC in futuro:
  // if (GET(Mb, c, i, j)) {
  //   SET(Sh_next, c, i, j, 0.0);
  //   SET(ST_next, c, i, j, 0.0);
  // }
}

// ----------------------------------------------------------------------------
// Riduzione CUDA (GLOBAL)
// ----------------------------------------------------------------------------

__global__
void reduceAdd_kernel(double *in, double *partial, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // carica il valore in shared memory
    if (gid < N)
        sdata[tid] = in[gid];
    else
        sdata[tid] = 0.0;

    __syncthreads();

    // riduzione nel blocco
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    // il thread 0 del blocco scrive il risultato parziale
    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}


double reduceAdd_gpu(int r, int c, double *buffer)
{
  int N = r * c;
  int threads = 256;
  int blocks  = (N + threads - 1) / threads;

  double *partial = nullptr;
  cudaCheck(cudaMallocManaged(&partial, blocks * sizeof(double)), "cudaMallocManaged(partial)");

  reduceAdd_kernel<<<blocks, threads, threads * sizeof(double)>>>(buffer, partial, N);
  cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize(reduceAdd_kernel)");

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

  // Domain boundaries and neighborhood
  int i_start = 0, i_end = sciara->domain->rows; // [i_start,i_end[: kernels application range along the rows
  int j_start = 0, j_end = sciara->domain->cols; // [j_start,j_end[: kernels application range along the cols

  int r = sciara->domain->rows;
  int c = sciara->domain->cols;

  dim3 blockDim(16, 16);
  dim3 gridDim((c + blockDim.x - 1) / blockDim.x,
               (r + blockDim.y - 1) / blockDim.y);

  // simulation initialization and loop
  double total_current_lava = -1.0;
  simulationInitialize(sciara);

  util::Timer cl_timer;

  int reduceInterval       = atoi(argv[REDUCE_INTERVL_ID]);
  double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

  while ((max_steps > 0 && sciara->simulation->step < max_steps) ||
         (sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
         (total_current_lava == -1.0 || total_current_lava > thickness_threshold))
  {
    sciara->simulation->elapsed_time += sciara->parameters->Pclock;
    sciara->simulation->step++;

    // ---------------- emitLava (host) ----------------
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        emitLava(i, j,
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
                 sciara->substates->ST_next);

    // Copia Sh_next -> Sh, ST_next -> ST (Unified Memory: copia host-host)
    std::memcpy(sciara->substates->Sh,
                sciara->substates->Sh_next,
                sizeof(double) * r * c);
    std::memcpy(sciara->substates->ST,
                sciara->substates->ST_next,
                sizeof(double) * r * c);

    // ---------------- computeOutflows (GPU) ----------------
    computeOutflows_kernel<<<gridDim, blockDim>>>(
        r, c,
        sciara->X->Xi,
        sciara->X->Xj,
        sciara->substates->Sz,
        sciara->substates->Sh,
        sciara->substates->ST,
        sciara->substates->Mf,
        sciara->parameters->Pc,
        sciara->parameters->a,
        sciara->parameters->b,
        sciara->parameters->c,
        sciara->parameters->d);
    cudaCheck(cudaDeviceSynchronize(), "computeOutflows_kernel");

    // ---------------- massBalance (GPU) ----------------
    massBalance_kernel<<<gridDim, blockDim>>>(
        r, c,
        sciara->X->Xi,
        sciara->X->Xj,
        sciara->substates->Sh,
        sciara->substates->Sh_next,
        sciara->substates->ST,
        sciara->substates->ST_next,
        sciara->substates->Mf);
    cudaCheck(cudaDeviceSynchronize(), "massBalance_kernel");

    std::memcpy(sciara->substates->Sh,
                sciara->substates->Sh_next,
                sizeof(double) * r * c);
    std::memcpy(sciara->substates->ST,
                sciara->substates->ST_next,
                sizeof(double) * r * c);

    // ---------------- computeNewTemperatureAndSolidification (GPU) ----------------
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
        sciara->substates->Mb);
    cudaCheck(cudaDeviceSynchronize(), "computeNewTemperatureAndSolidification_kernel");

    std::memcpy(sciara->substates->Sz,
                sciara->substates->Sz_next,
                sizeof(double) * r * c);
    std::memcpy(sciara->substates->Sh,
                sciara->substates->Sh_next,
                sizeof(double) * r * c);
    std::memcpy(sciara->substates->ST,
                sciara->substates->ST_next,
                sizeof(double) * r * c);

    // ---------------- boundaryConditions (GPU, attualmente NOP) ----------------
    boundaryConditions_kernel<<<gridDim, blockDim>>>(
        r, c,
        sciara->substates->Mf,
        sciara->substates->Mb,
        sciara->substates->Sh,
        sciara->substates->Sh_next,
        sciara->substates->ST,
        sciara->substates->ST_next);
    cudaCheck(cudaDeviceSynchronize(), "boundaryConditions_kernel");

    std::memcpy(sciara->substates->Sh,
                sciara->substates->Sh_next,
                sizeof(double) * r * c);
    std::memcpy(sciara->substates->ST,
                sciara->substates->ST_next,
                sizeof(double) * r * c);

    // ---------------- Global reduction (GPU) ----------------
    if (sciara->simulation->step % reduceInterval == 0)
      total_current_lava = reduceAdd_gpu(r, c, sciara->substates->Sh);
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Step %d\n", sciara->simulation->step);
  printf("Elapsed time [s]: %lf\n", cl_time);
  printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
  printf("Current lava [m]: %lf\n", total_current_lava);

  printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
  saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

  printf("Releasing memory...\n");
  finalize(sciara);

  return 0;
}
