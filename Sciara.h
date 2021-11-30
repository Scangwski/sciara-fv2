#ifndef CA_H_
#define CA_H_

#include "GISInfo.h"
#include "vent.h"
#include <math.h>
#include <stdlib.h>

#define NUMBER_OF_OUTFLOWS 8

#define MIN_ALG		0
#define PROP_ALG	1

typedef struct
{
	double* Sz;		    //Altitude
  double* Sz_next;
	double* Slt;	    //Lava thickness
  double* Slt_next;
	double* St;		    //Lava temperature
  double* St_next;
	double* Sf;		    //Flows Substates
	int*    Mv;		    //Matrix of the vents
	bool*   Mb;		    //Matrix of the topography bound
	double* Msl;	    //Matrix of the solidified lava

} Substates;

typedef struct
{
	int    maximum_steps;	//... go for maximum_steps steps (0 for loop)
	double stopping_threshold;	//se negativa non si effettua il controllo sulla pausa
	int    refreshing_step;	//I thread grafici vengono avviati ogni repaint_step passi
	double thickness_visual_threshold;	//in LCMorphology viene disegnato nero solo se mMD > visual_threshold
	double Pclock;	//AC clock [s]
	double Pc;		//cell side
	double Pac;		//area of the cell
	double PTsol;	//temperature of solidification
	double PTvent;	//temperature of lava at vent
	double Pr_Tsol;
	double Pr_Tvent;
	double a;		// parametro per calcolo Pr
	double b;		// parametro per calcolo Pr
	double Phc_Tsol;
	double Phc_Tvent;
	double c;		// parametro per calcolo hc
	double d;		// parametro per calcolo hc
	double Pcool;
	double Prho;	//density
	double Pepsilon;	//emissivity
	double Psigma;	//Stephen-Boltzamnn constant
	double Pcv;		//Specific heat
	int algorithm;	
	int rows;
	int cols;
	double rad2;
	unsigned int emission_time;
	vector<TEmissionRate> emission_rate;
	vector<TVent> vent;
	double elapsed_time; //tempo trascorso dall'inizio della simulazione [s]
  int step;

	double effusion_duration;
	Substates * substates;

} Sciara;

// ----------------------------------------------------------------------------
// Memory allocation function for 2D linearized buffers
// ----------------------------------------------------------------------------
void allocateSubstates(Sciara *sciara);
void deallocateSubstates(Sciara *sciara);


// bool** MakeBorder(int ly, int lx, bool **Border, double** mQ)
// {
//   int x, y;
// 
//   //prima riga
//   y = 0;
//   for (x = 0; x < lx; x++ )
//     if (mQ[x][y] >= 0)
//       Border[x][y] = true;
//   //ultima riga
//   y = ly - 1;
//   for (x = 0; x < lx; x++ )
//     if (mQ[x][y] >= 0)
//       Border[x][y] = true;
//   //prima colonna x = 0;
//   for (y = 0; y < ly; y++ )
//     if (mQ[x][y] >= 0)
//       Border[x][y] = true;
//   //ultima colonna
//   x = lx - 1;
//   for (y = 0; y < ly; y++ )
//     if (mQ[x][y] >= 0)
//       Border[x][y] = true;
//   //il resto
//   for (int x = 1; x < lx-1; x++)
//     for (int y = 1; y < ly-1; y++)
//       if (mQ[x][y] >= 0)
//       {
//         mooreNeighbors(x, y, neigh);
//         for (int i = 1; i < MOORE_NEIGHBORS; i++ )
//           if (mQ[neigh[i].x][neigh[i].y] < 0)
//           {
//             Border[neigh[0].x][neigh[0].y] = true;
//             break;
//           }
//       }
// 
//   return Border;
// }

#endif /* CA_H_ */
