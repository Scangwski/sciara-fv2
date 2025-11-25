#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <cuda_runtime.h>
#include <cuda.h>

extern "C" {
    void emitLavaCUDA(double*,double*,double*,int,int,
                      int*,int*,double*,int,double);

    void computeOutflowsCUDA(double*,double*,double*,double*,int,int,
                             double,double,double,double);

    void massBalanceCUDA(double*,double*,double*,double*,double*,int,int);

    void computeNewTempAndSolidCUDA(double,double,double,double,double,double,double,double,
                                    double*,double*,double*,double*,double*,double*,double*,bool*,
                                    int,int);

    void boundaryConditionsCUDA();

    double reduceAddCUDA(double*,int,int);
}

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
// computing kernels, aka elementary processes in the XCA terminology
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
  for (int k = 0; k < vent.size(); k++)
    if (i == vent[k].y() && j == vent[k].x())
    {
      SET(Sh_next, c, i, j, GET(Sh, c, i, j) + vent[k].thickness(elapsed_time, Pclock, emission_time, Pac));
      SET(ST_next, c, i, j, PTvent);

      total_emitted_lava += vent[k].thickness(elapsed_time, Pclock, emission_time, Pac);
    }
}

void computeOutflows(
    int i,
    int j,
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
  bool eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];
  double w[MOORE_NEIGHBORS];  // Distances between central and adjecent cells
  double Pr[MOORE_NEIGHBORS]; // Relaiation rate arraj
  double f[MOORE_NEIGHBORS];
  bool loop;
  int counter;
  double sz0, sz, T, avg, rr, hc;

  if (GET(Sh, c, i, j) <= 0)
    return;

  T = GET(ST, c, i, j);
  rr = pow(10, _a + _b * T);
  hc = pow(10, _c + _d * T);

  for (int k = 0; k < MOORE_NEIGHBORS; k++)
  {
    sz0 = GET(Sz, c, i, j);
    sz = GET(Sz, c, i + Xi[k], j + Xj[k]);
    h[k] = GET(Sh, c, i + Xi[k], j + Xj[k]);
    w[k] = Pc;
    Pr[k] = rr;

    if (k < VON_NEUMANN_NEIGHBORS)
      z[k] = sz;
    else
      z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
  }

  H[0] = z[0];
  theta[0] = 0;
  eliminated[0] = false;
  for (int k = 1; k < MOORE_NEIGHBORS; k++)
    if (z[0] + h[0] > z[k] + h[k])
    {
      H[k] = z[k] + h[k];
      theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
      eliminated[k] = false;
    }
    else
    {
      // H[k] = 0;
      // theta[k] = 0;
      eliminated[k] = true;
    }

  do
  {
    loop = false;
    avg = h[0];
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

void massBalance(
    int i,
    int j,
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
  const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};
  double inFlow;
  double outFlow;
  double neigh_t;
  double initial_h = GET(Sh, c, i, j);
  double initial_t = GET(ST, c, i, j);
  double h_next = initial_h;
  double t_next = initial_h * initial_t;

  for (int n = 1; n < MOORE_NEIGHBORS; n++)
  {
    neigh_t = GET(ST, c, i + Xi[n], j + Xj[n]);
    inFlow = BUF_GET(Mf, r, c, inflowsIndices[n - 1], i + Xi[n], j + Xj[n]);

    outFlow = BUF_GET(Mf, r, c, n - 1, i, j);

    h_next += inFlow - outFlow;
    t_next += (inFlow * neigh_t - outFlow * initial_t);
  }

  if (h_next > 0)
  {
    t_next /= h_next;
    SET(ST_next, c, i, j, t_next);
    SET(Sh_next, c, i, j, h_next);
  }
}

void computeNewTemperatureAndSolidification(
    int i,
    int j,
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
    bool *Mb)
{
  double nT, aus;
  double z = GET(Sz, c, i, j);
  double h = GET(Sh, c, i, j);
  double T = GET(ST, c, i, j);

  if (h > 0 && GET(Mb, c, i, j) == false)
  {
    aus = 1.0 + (3 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
    nT = T / pow(aus, 1.0 / 3.0);

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

void boundaryConditions(int i, int j,
                        int r,
                        int c,
                        double *Mf,
                        bool *Mb,
                        double *Sh,
                        double *Sh_next,
                        double *ST,
                        double *ST_next)
{
  return;
  if (GET(Mb, c, i, j))
  {
    SET(Sh_next, c, i, j, 0.0);
    SET(ST_next, c, i, j, 0.0);
  }
}

double reduceAdd(int r, int c, double *buffer)
{
  double sum = 0.0;
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++)
      sum += GET(buffer, c, i, j);

  return sum;
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Sciara *sciara;
    init(sciara);

    // Load input data
    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;
    size_t bytes = rows * cols * sizeof(double);

    simulationInitialize(sciara);

    util::Timer cl_timer;

    int reduceInterval      = atoi(argv[REDUCE_INTERVL_ID]);
    double threshold        = atof(argv[THICKNESS_THRESHOLD_ID]);

    // ------------------------------
    // PREPARAZIONE VENTS PER CUDA
    // ------------------------------
    int nVents = sciara->simulation->vent.size();

    int *d_ventX = nullptr;
    int *d_ventY = nullptr;
    double *d_ventThick = nullptr;

    if (nVents > 0)
    {
        cudaMallocManaged(&d_ventX,     nVents * sizeof(int));
        cudaMallocManaged(&d_ventY,     nVents * sizeof(int));
        cudaMallocManaged(&d_ventThick, nVents * sizeof(double));

        for (int k = 0; k < nVents; k++)
        {
            d_ventX[k] = sciara->simulation->vent[k].x();
            d_ventY[k] = sciara->simulation->vent[k].y();
        }
    }

    // ------------------------------
    // LOOP DELLA SIMULAZIONE CUDA
    // ------------------------------
    double total_current_lava = -1;

    while ( (max_steps > 0 && sciara->simulation->step < max_steps) ||
            (sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
            (total_current_lava == -1 || total_current_lava > threshold) )
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        // ----------------------------------------------------
        // 1) EMIT LAVA
        // ----------------------------------------------------
        if (nVents > 0)
        {
            for (int k = 0; k < nVents; k++)
            {
                double th = sciara->simulation->vent[k].thickness(
                    sciara->simulation->elapsed_time,
                    sciara->parameters->Pclock,
                    sciara->simulation->emission_time,
                    sciara->parameters->Pac
                );
                d_ventThick[k] = th;
                sciara->simulation->total_emitted_lava += th;
            }
        }

        emitLavaCUDA(
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST_next,
            rows, cols,
            d_ventX, d_ventY, d_ventThick, nVents,
            sciara->parameters->PTvent
        );

        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, bytes, cudaMemcpyDeviceToDevice);

        // ----------------------------------------------------
        // 2) COMPUTE OUTFLOWS
        // ----------------------------------------------------
        computeOutflowsCUDA(
    sciara->substates->Sz,
    sciara->substates->Sh,
    sciara->substates->ST,
    sciara->substates->Mf,
    rows,
    cols,
    sciara->parameters->Pc,
    sciara->parameters->a,
    sciara->parameters->b,
    sciara->parameters->c,
    sciara->parameters->d
);



        // ----------------------------------------------------
        // 3) MASS BALANCE
        // ----------------------------------------------------
        massBalanceCUDA(
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mf,
            rows, cols
        );

        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, bytes, cudaMemcpyDeviceToDevice);

        // ----------------------------------------------------
        // 4) TEMPERATURE & SOLIDIFICATION
        // ----------------------------------------------------
        computeNewTempAndSolidCUDA(
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
            sciara->substates->Mhs,
            sciara->substates->Mb,
            rows, cols
        );

        cudaMemcpy(sciara->substates->Sz, sciara->substates->Sz_next, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, bytes, cudaMemcpyDeviceToDevice);

        // ----------------------------------------------------
        // 5) BOUNDARY CONDITIONS
        // ----------------------------------------------------
        boundaryConditionsCUDA();

        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, bytes, cudaMemcpyDeviceToDevice);

        // ----------------------------------------------------
        // 6) GLOBAL REDUCTION
        // ----------------------------------------------------
        if (sciara->simulation->step % reduceInterval == 0)
            total_current_lava = reduceAddCUDA(
                sciara->substates->Sh,
                rows, cols
            );
    }

    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Step %d\n", sciara->simulation->step);
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

    if (d_ventX)     cudaFree(d_ventX);
    if (d_ventY)     cudaFree(d_ventY);
    if (d_ventThick) cudaFree(d_ventThick);

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}
