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

void simulationInit (
    int i, 
    int j, 
    int r, 
    int c, 
    double* Sz, 
    double* Sz_next, 
    double* Slt, 
    double* Slt_next, 
    double* St, 
    double* St_next, 
    double* Sf, 
    bool* Mb)
{
  SET(Sz_next, c,i,j,GET(Sz,c,i,j) ); 
  SET(Slt_next,c,i,j,GET(Slt,c,i,j) ); 
  SET(St_next, c,i,j,GET(St,c,i,j) ); 
  SET(Mb,      c,i,j,false); 
  for (int n=0; n<NUMBER_OF_OUTFLOWS; n++)
    BUF_SET(Sf,r,c,n,i,j,0.0); 
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------

void update(
    int i,
    int j,
    int r,
    int c,
    double* S,
    double* S_next)
{
  SET(S,c,i,j,GET(S_next,c,i,j)); // S[i][j]=S_next[i][j];
}

void emitLava(
    int i,
    int j,
    int r,
    int c,
    vector<TVent> &vent, 
    double elapsed_time,
    double Pclock,
    double emission_time,
    double Pac,
    double PTvent,
    double* Slt,
    double* Slt_next,
    double* St_next)
{
  double emitted_lava = 0;
  
  for (int k = 0; k < vent.size(); k++)
  {
    int xVent = vent[k].x();
    int yVent = vent[k].y();

    if (i == yVent && j == xVent) 
    {
      emitted_lava = vent[k].thickness(elapsed_time, Pclock, emission_time, Pac);
      if (emitted_lava > 0)
      {
        double h_new = GET(Slt, c, yVent, xVent) + emitted_lava;
        SET(Slt_next, c, yVent, xVent, h_new);
        SET(St_next,  c, yVent, xVent, PTvent); 
      }
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
    if (f[k] > 0) 
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
  int outFlowsIndexes[NUMBER_OF_OUTFLOWS] = { 3, 2, 1, 0, 6, 7, 4, 5 };
  int n;
  double initial_h = GET(Slt,c,i,j);
  double initial_t = GET(St,c,i,j);
  double residualTemperature = initial_h * initial_t;
  double residualLava = initial_h;
  double h_next = initial_h;
  double t_next;

  double ht = 0;
  double inSum = 0;
  double outSum = 0;

  for (n = 1; n < MOORE_NEIGHBORS; n++)
  {
    double inFlow = BUF_GET(Sf,r,c,outFlowsIndexes[n-1],i+Xi[n],j+Xj[n]);
    double outFlow = BUF_GET(Sf,r,c,n-1,i,j);
    double neigh_t = GET(St,c,i+Xi[n],j+Xj[n]);
    ht += inFlow * neigh_t;
    inSum += inFlow;
    outSum += outFlow;
  }
  h_next += inSum - outSum;
  SET(Slt_next,c,i,j,h_next);

  if (inSum > 0 || outSum > 0) {
    residualLava -= outSum;
    t_next = (residualLava * initial_t + ht) / (residualLava + inSum);
    SET(St_next,c,i,j,t_next);
  }
}

void resetFlowsAndBoundaries (int i, int j, 
            int r, 
            int c, 
            double* Sf,
            bool*   Mb,
            double* Slt,
            double* Slt_next,
            double* St,
            double* St_next)
{
  for (int n=0; n<NUMBER_OF_OUTFLOWS; n++)
    BUF_SET(Sf,r,c,n,i,j,0.0);

  if (GET(Mb,c,i,j))
  {
    SET(Slt,     c,i,j,0.0);
    SET(Slt_next,c,i,j,0.0);
    SET(St,      c,i,j,0.0);
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
  double nT, h, T, aus;
  double sz = GET(Sz,c,i,j); //calGet2Dr(model, sciara->substates->Sz, i, j);
  double sh = GET(Slt,c,i,j); //calGet2Dr(model, sciara->substates->Slt, i, j);
  double st = GET(St,c,i,j); //calGet2Dr(model, sciara->substates->St, i, j);

  if (sh > 0 && GET(Mb,c,i,j) == false ) 
  {
    h = sh;
    T = st;
    if (h != 0) 
    {
      nT = T;

      aus = 1.0 + (3 * pow(nT, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
      st = nT / pow(aus, 1.0 / 3.0);
      SET(St_next,c,i,j,st); //calSet2Dr(model, sciara->substates->St, i, j, st);
    }

    //solidification
    if (st <= PTsol && sh > 0)
    {
      SET(Sz_next,c,i,j,sz+sh); //calSet2Dr(model, sciara->substates->Sz, i, j, sz + sh);
      SET(Msl,c,i,j, GET(Msl,c,i,j)+sh); //calSetCurrent2Dr(model, sciara->substates->Msl, i, j, calGet2Dr(model, sciara->substates->Msl, i, j) + sh);
      SET(Slt_next,c,i,j,0.0); //calSet2Dr(model, sciara->substates->Slt, i, j, 0);
      SET(St_next,c,i,j,PTsol); //calSet2Dr(model, sciara->substates->St, i, j, sciara->PTsol);
    } 
    else
      SET(Sz_next,c,i,j, sz); // calSet2Dr(model, sciara->substates->Sz, i, j, sz);
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
  // The adopted von Neuman neighborhood
  // Format: flow_index:cell_label:(row_index,col_index)
  //
  //   cell_label in [0,1,2,3,4,5,6,7,8]: label assigned to each cell in the neighborhood
  //   flow_index in   [0,1,2,3,4,5,6,7]: outgoing flow indices in Sf from cell 0 to the others
  //       (row_index,col_index): 2D relative indices of the cells
  //
  //
  //    cells               cells         outflows
  //    coordinates         labels        indices
  //
  //   -1,-1|-1,0| 1,1      |5|1|8|       |4|0|7|
  //    0,-1| 0,0| 0,1      |2|0|3|       |1| |2|
  //    1,-1| 1,0|-1,1      |6|4|7|       |5|3|6|

  Sciara *sciara;
  init(sciara);

  // Input data 
  int max_steps = atoi(argv[MAX_STEPS_ID]);
  loadConfiguration(argv[INPUT_PATH_ID], sciara);

  SimulationInitialize(sciara);

  // Domain boundaries and neighborhood
  int i_start = 0, i_end = sciara->domain->rows-1;        // [i_start,i_end[: kernels application range along the rows
  int j_start = 0, j_end = sciara->domain->cols-1;        // [i_start,i_end[: kernels application range along the rows
  // int Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; // Xj: Moore neighborhood row coordinates (see below)
  // int Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; // Xj: Moore neighborhood col coordinates (see below)

  // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
#pragma omp parallel for
  for (int i = i_start; i < i_end; i++)
    for (int j = j_start; j < j_end; j++)
      simulationInit(i, j, 
          sciara->domain->rows, 
          sciara->domain->cols, 
          sciara->substates->Sz, 
          sciara->substates->Sz_next, 
          sciara->substates->Slt, 
          sciara->substates->Slt_next, 
          sciara->substates->St, 
          sciara->substates->St_next, 
          sciara->substates->Sf, 
          sciara->substates->Mb);

  util::Timer cl_timer;
  // simulation loop
  while ( (max_steps > 0 && sciara->simulation->step < max_steps) 
     /*|| sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration */)
  {
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
            sciara->parameters->Pac, 
            sciara->parameters->PTvent, 
            sciara->substates->Slt, 
            sciara->substates->Slt_next,
            sciara->substates->St_next);
    calCopyBuffer2Dr(sciara->substates->Slt_next, sciara->substates->Slt, sciara->domain->rows, sciara->domain->cols);
    calCopyBuffer2Dr(sciara->substates->St_next,  sciara->substates->St,  sciara->domain->rows, sciara->domain->cols);
    //swap_pointers(sciara->substates->Slt_next, sciara->substates->Slt);
    //swap_pointers(sciara->substates->St_next,  sciara->substates->St);

    // Apply the computeOutflows kernel to the whole domain
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        computeOutflows(i, j, 
            sciara->domain->rows, 
            sciara->domain->cols, 
            Xi, 
            Xj,
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
        massBalance (i, j, 
            sciara->domain->rows, 
            sciara->domain->cols, 
            Xi, 
            Xj, 
            sciara->substates->Slt, 
            sciara->substates->Slt_next, 
            sciara->substates->St, 
            sciara->substates->St_next, 
            sciara->substates->Sf);
    calCopyBuffer2Dr(sciara->substates->Slt_next, sciara->substates->Slt, sciara->domain->rows, sciara->domain->cols);
    calCopyBuffer2Dr(sciara->substates->St_next,  sciara->substates->St,  sciara->domain->rows, sciara->domain->cols);
    //swap_pointers(sciara->substates->Slt_next, sciara->substates->Slt);
    //swap_pointers(sciara->substates->St_next,  sciara->substates->St);

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
    calCopyBuffer2Dr(sciara->substates->Sz_next,  sciara->substates->Sz,  sciara->domain->rows, sciara->domain->cols);
    calCopyBuffer2Dr(sciara->substates->Slt_next, sciara->substates->Slt, sciara->domain->rows, sciara->domain->cols);
    calCopyBuffer2Dr(sciara->substates->St_next,  sciara->substates->St,  sciara->domain->rows, sciara->domain->cols);
    //swap_pointers(sciara->substates->Sz_next,  sciara->substates->Sz);
    //swap_pointers(sciara->substates->Slt_next, sciara->substates->Slt);
    //swap_pointers(sciara->substates->St_next,  sciara->substates->St);

    // Apply the resetFlows kernel to the whole domain and update the Slt and St state variables
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        resetFlowsAndBoundaries (i, j, 
            sciara->domain->rows, 
            sciara->domain->cols, 
            sciara->substates->Sf,
            sciara->substates->Mb,
            sciara->substates->Slt,
            sciara->substates->Slt_next,
            sciara->substates->St,
            sciara->substates->St_next);


    sciara->simulation->elapsed_time += sciara->parameters->Pclock;
    sciara->simulation->step++;
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
  saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

  printf("Releasing memory...\n");
  //deallocateSubstates(sciara);
  //delete sciara;
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
