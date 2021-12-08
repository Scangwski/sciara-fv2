#include "cal2DBuffer.h"
#include "configurationPathLib.h"
#include "GISInfo.h"
#include "io.h"
#include "vent.h"
#include <omp.h>
#include <new>
#include "Sciara.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.hpp"

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------

#define INPUT_PATH_ID  1
#define OUTPUT_PATH_ID 2
#define MAX_STEPS_ID   3

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------

#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------

void evaluatePowerLawParams(double PTvent, double PTsol, double value_sol, double value_vent, double &k1, double &k2)
{
	k2 = ( log10(value_vent) - log10(value_sol) ) / (PTvent - PTsol) ;
	k1 = log10(value_sol) - k2*(PTsol);
}

void SimulationInitialize(Sciara* sciara)
{
  //dichiarazioni
  unsigned int maximum_number_of_emissions = 0;

  //azzeramento dello step dell'AC
  sciara->simulation->step = 0;
  sciara->simulation->elapsed_time = 0;

  //determinazione numero massimo di passi
  for (unsigned int i = 0; i < sciara->simulation->emission_rate.size(); i++)
    if (maximum_number_of_emissions < sciara->simulation->emission_rate[i].size())
      maximum_number_of_emissions = sciara->simulation->emission_rate[i].size();
  //maximum_steps_from_emissions = (int)(emission_time/Pclock*maximum_number_of_emissions);
  sciara->simulation->effusion_duration = sciara->simulation->emission_time * maximum_number_of_emissions;
  sciara->simulation->total_emitted_lava = 0;

  //definisce il bordo della morfologia
  MakeBorder(sciara);

  //calcolo a b (parametri viscosità) c d (parametri resistenza al taglio)
  evaluatePowerLawParams(
      sciara->parameters->PTvent, 
      sciara->parameters->PTsol, 
      sciara->parameters->Pr_Tsol,  
      sciara->parameters->Pr_Tvent,  
      sciara->parameters->a, 
      sciara->parameters->b);
  evaluatePowerLawParams(
      sciara->parameters->PTvent,
      sciara->parameters->PTsol,
      sciara->parameters->Phc_Tsol,
      sciara->parameters->Phc_Tvent,
      sciara->parameters->c,
      sciara->parameters->d);
}

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
    double& total_emitted_lava,
    double Pac,
    double PTvent,
    double* Slt,
    double* Slt_next,
    double* St,
    double* St_next)
{
  int xVent;
  int yVent;
  double current_lava;
  double emitted_lava;
  double next_lava;
  double current_temperature;

  for (int k = 0; k < vent.size(); k++)
  {
    xVent = vent[k].x();
    yVent = vent[k].y();

    if (i == yVent && j == xVent) 
    {
      current_lava = GET(Slt, c, i, j);
      emitted_lava = vent[k].thickness(elapsed_time, Pclock, emission_time, Pac);
      next_lava = current_lava + emitted_lava;
      total_emitted_lava += emitted_lava;
      
      SET(Slt_next, c, i, j, next_lava);
      SET(St_next,  c, i, j, PTvent); 
    }
  }
}


double powerLaw(double k1, double k2, double T)
{
  double log_value = k1 + k2 * T;
  return pow(10, log_value);
}

void computeOutflows (
    int i, 
    int j, 
    int r, 
    int c, 
    int* Xi, 
    int* Xj, 
    double *Sz, 
    double *Slt, 
    double *St,
    double *Sf, 
    double  Pc, 
    double  _a,
    double  _b,
    double  _c,
    double  _d)
{
  bool n_eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];
  double w[MOORE_NEIGHBORS];		//Distances between central and adjecent cells
  double Pr[MOORE_NEIGHBORS];		//Relaiation rate arraj
  bool loop;
  int counter;
  double avg, _w, _Pr, hc, sum, sumZ;

  double	f[MOORE_NEIGHBORS];

  double t = GET(St, c, i, j);

  if (GET(Slt,c,i,j) <=0)
    return;

  _w = Pc;
  _Pr = powerLaw(_a, _b, t);
  hc = powerLaw(_c, _d, t);
  for (int k = 0; k < MOORE_NEIGHBORS; k++)
  {
    h[k] = GET(Slt, c, i+Xi[k], j+Xj[k]);
    H[k] = f[k] = theta[k] = 0;
    w[k] = _w;
    Pr[k] = _Pr;
    double sz = GET(Sz, c, i+Xi[k], j+Xj[k]);
    double sz0 = GET(Sz, c, i, j);

    if (k < VON_NEUMANN_NEIGHBORS)
      z[k] = GET(Sz, c, i+Xi[k], j+Xj[k]);
    else
      z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
  }

  H[0] = z[0];
  n_eliminated[0] = true;

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
    if (z[0] + h[0] > z[k] + h[k])
    {
      H[k] = z[k] + h[k];
      theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
      n_eliminated[k] = true;
    } 
    else
      n_eliminated[k] = false;

  do {
    loop = false;
    avg = h[0];
    counter = 0;
    for (int k = 0; k < MOORE_NEIGHBORS; k++)
      if (n_eliminated[k]) {
        avg += H[k];
        counter++;
      }
    if (counter != 0)
      avg = avg / double(counter);
    for (int k = 0; k < MOORE_NEIGHBORS; k++)
      if (n_eliminated[k] && avg <= H[k]) {
        n_eliminated[k] = false;
        loop = true;
      }
  } while (loop);

  for (int k = 1; k < MOORE_NEIGHBORS; k++) 
    if (n_eliminated[k] && h[0] > hc * cos(theta[k]))
      f[k] = Pr[k] * (avg - H[k]);

  for (int k = 1; k < MOORE_NEIGHBORS; k++)
    BUF_SET(Sf,r,c,k-1,i,j,f[k]);
}

void massBalance(
    int i, 
    int j, 
    int r, 
    int c, 
    int* Xi, 
    int* Xj, 
    double *Slt, 
    double *Slt_next, 
    double *St,
    double *St_next,
    double *Sf)
{
  const int inflowsIndices[NUMBER_OF_OUTFLOWS] = { 3, 2, 1, 0, 6, 7, 4, 5 };
  double inFlow;
  double outFlow;
  double neigh_t;
  double initial_h = GET(Slt,c,i,j);
  double initial_t = GET(St,c,i,j);
  double h_next = initial_h;
  double t_next = initial_h * initial_t;

  for (int n = 1; n < MOORE_NEIGHBORS; n++)
  {
    neigh_t = GET(St,c,i+Xi[n],j+Xj[n]);
    inFlow  = BUF_GET(Sf,r,c,inflowsIndices[n-1],i+Xi[n],j+Xj[n]);

    outFlow = BUF_GET(Sf,r,c,n-1,i,j);

    h_next +=  inFlow - outFlow;
    t_next += (inFlow * neigh_t - outFlow * initial_t);
  }

  if (h_next > 0)
  {
    t_next /= h_next;
    SET(St_next,c,i,j,t_next);
    SET(Slt_next,c,i,j,h_next);
  }
}

void boundaryConditions (int i, int j, 
            int r, 
            int c, 
            double* Sf,
            bool*   Mb,
            double* Slt,
            double* Slt_next,
            double* St,
            double* St_next)
{
  return;
  if (GET(Mb,c,i,j))
  {
    SET(Slt_next,c,i,j,0.0);
    SET(St_next, c,i,j,0.0);
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
    double *Slt, 
    double *Slt_next, 
    double *St,
    double *St_next,
    double *Sf,
    double *Msl,
    bool   *Mb)
{
  double nT, aus;
  double z = GET(Sz,c,i,j);
  double h = GET(Slt,c,i,j);
  double T = GET(St,c,i,j);

  if (h > 0 && GET(Mb,c,i,j) == false ) 
  {
    aus = 1.0 + (3 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
    nT = T / pow(aus, 1.0 / 3.0);

    if (nT > PTsol) // no solidification
      SET(St_next,c,i,j, nT);
    else            // solidification
    {
      SET(Sz_next,c,i,j,z+h);
      SET(Slt_next,c,i,j,0.0);
      SET(St_next,c,i,j,PTsol);
      SET(Msl,c,i,j, GET(Msl,c,i,j)+h);
    }

  }
}


template <typename type>
void swap_pointers(type*& p1, type*& p2)
{
  type *pt = p1;
  p1 = p2;
  p2 = pt;
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
  SimulationInitialize(sciara);

  // Domain boundaries and neighborhood
  int i_start = 0, i_end = sciara->domain->rows-1;        // [i_start,i_end[: kernels application range along the rows
  int j_start = 0, j_end = sciara->domain->cols-1;        // [i_start,i_end[: kernels application range along the rows

  double total_current_lava = 0.0;
  double total_outflows = 0.0;

  util::Timer cl_timer;
  // simulation loop
  while ( (max_steps > 0 && sciara->simulation->step < max_steps)  &&  (sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) )
  {
    sciara->simulation->elapsed_time += sciara->parameters->Pclock;
    sciara->simulation->step++;


    // Apply the emitLava kernel to the whole domain and update the Slt and St state variables
#pragma omp parallel for
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
            sciara->substates->Slt, 
            sciara->substates->Slt_next,
            sciara->substates->St, 
            sciara->substates->St_next);
    //swap_pointers(sciara->substates->Slt_next, sciara->substates->Slt);
    //swap_pointers(sciara->substates->St_next,  sciara->substates->St);
    memcpy(sciara->substates->Slt, sciara->substates->Slt_next, sizeof(double)*sciara->domain->rows*sciara->domain->cols);
    memcpy(sciara->substates->St,  sciara->substates->St_next,  sizeof(double)*sciara->domain->rows*sciara->domain->cols);


    // Apply the computeOutflows kernel to the whole domain
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        computeOutflows(i, j, 
            sciara->domain->rows, 
            sciara->domain->cols, 
            sciara->X->Xi, 
            sciara->X->Xj, 
            sciara->substates->Sz, 
            sciara->substates->Slt, 
            sciara->substates->St, 
            sciara->substates->Sf, 
            sciara->parameters->Pc, 
            sciara->parameters->a, 
            sciara->parameters->b, 
            sciara->parameters->c, 
            sciara->parameters->d);


    // Apply the massBalance mass balance kernel to the whole domain and update the Slt and St state variables
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        massBalance(i, j,
            sciara->domain->rows, 
            sciara->domain->cols, 
            sciara->X->Xi, 
            sciara->X->Xj, 
            sciara->substates->Slt, 
            sciara->substates->Slt_next, 
            sciara->substates->St, 
            sciara->substates->St_next, 
            sciara->substates->Sf);
    //swap_pointers(sciara->substates->Slt_next, sciara->substates->Slt);
    //swap_pointers(sciara->substates->St_next,  sciara->substates->St);
    memcpy(sciara->substates->Slt, sciara->substates->Slt_next, sizeof(double)*sciara->domain->rows*sciara->domain->cols);
    memcpy(sciara->substates->St,  sciara->substates->St_next,  sizeof(double)*sciara->domain->rows*sciara->domain->cols);


    // Apply the computeNewTemperatureAndSolidification kernel to the whole domain
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        computeNewTemperatureAndSolidification (i, j, 
            sciara->domain->rows, 
            sciara->domain->cols, 
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
            sciara->substates->Slt, 
            sciara->substates->Slt_next,
            sciara->substates->St, 
            sciara->substates->St_next,
            sciara->substates->Sf, 
            sciara->substates->Msl, 
            sciara->substates->Mb);
    //swap_pointers(sciara->substates->Sz_next,  sciara->substates->Sz);
    //swap_pointers(sciara->substates->Slt_next, sciara->substates->Slt);
    //swap_pointers(sciara->substates->St_next,  sciara->substates->St);
    memcpy(sciara->substates->Sz,  sciara->substates->Sz_next,  sizeof(double)*sciara->domain->rows*sciara->domain->cols);
    memcpy(sciara->substates->Slt, sciara->substates->Slt_next, sizeof(double)*sciara->domain->rows*sciara->domain->cols);
    memcpy(sciara->substates->St,  sciara->substates->St_next,  sizeof(double)*sciara->domain->rows*sciara->domain->cols);


    // Apply the boundaryConditions kernel to the whole domain and update the Slt and St state variables
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        boundaryConditions (i, j, 
            sciara->domain->rows, 
            sciara->domain->cols, 
            sciara->substates->Sf,
            sciara->substates->Mb,
            sciara->substates->Slt,
            sciara->substates->Slt_next,
            sciara->substates->St,
            sciara->substates->St_next);
    memcpy(sciara->substates->Slt, sciara->substates->Slt_next, sizeof(double)*sciara->domain->rows*sciara->domain->cols);
    memcpy(sciara->substates->St,  sciara->substates->St_next,  sizeof(double)*sciara->domain->rows*sciara->domain->cols);

  }


  // Compute the current amount of lava on the surface
  total_current_lava = 0.0;
  total_outflows = 0.0;
  for (int i = 0; i < sciara->domain->rows; i++)
    for (int j = 0; j < sciara->domain->cols; j++)
    {
      total_current_lava += GET(sciara->substates->Slt,sciara->domain->cols,i,j) + GET(sciara->substates->Msl,sciara->domain->cols,i,j);
      for (int n = 0; n < NUMBER_OF_OUTFLOWS; n++)
        total_outflows += BUF_GET(sciara->substates->Sf,sciara->domain->rows,sciara->domain->cols,n,i,j);
    }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);
  printf("Emitted lava: %lf [m]\n", sciara->simulation->total_emitted_lava);
  printf("Current lava: %lf [m]\n", total_current_lava);
  printf("Current outf: %lf [m]\n", total_outflows);

  printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
  saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

  printf("Releasing memory...\n");
  finalize(sciara);

  return 0;
}

//============================================================================

// void steering() {
//   for (int i = 0; i < NUMBER_OF_OUTFLOWS; ++i)
//     calInitSubstate2Dr(model, sciara->substates->f[i], 0);
// 
//   for (int i = 0; i < sciara->domain->rows; i++)
//     for (int j = 0; j < sciara->domain->cols; j++)
//       if (calGet2Db(model, sciara->substates->Mb, i, j) == true) {
//         calSet2Dr(model, sciara->substates->Slt, i, j, 0);
//         calSet2Dr(model, sciara->substates->St, i, j, 0);
//       }
//   sciara->elapsed_time += sciara->Pclock;
//   //	updateVentsEmission(model);
// }
