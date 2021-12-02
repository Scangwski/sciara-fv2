#include "Sciara.h"
#include "cal2DBuffer.h"

void allocateSubstates(Sciara *sciara)
{
	sciara->substates->Sz       = new (std::nothrow) double[sciara->domain->rows*sciara->domain->cols];
  sciara->substates->Sz_next  = new (std::nothrow) double[sciara->domain->rows*sciara->domain->cols];
	sciara->substates->Slt      = new (std::nothrow) double[sciara->domain->rows*sciara->domain->cols];
  sciara->substates->Slt_next = new (std::nothrow) double[sciara->domain->rows*sciara->domain->cols];
	sciara->substates->St       = new (std::nothrow) double[sciara->domain->rows*sciara->domain->cols];
  sciara->substates->St_next  = new (std::nothrow) double[sciara->domain->rows*sciara->domain->cols];
	sciara->substates->Sf       = new (std::nothrow) double[sciara->domain->rows*sciara->domain->cols*NUMBER_OF_OUTFLOWS];
//sciara->substates->Mv       = new (std::nothrow)    int[sciara->domain->rows*sciara->domain->cols];
	sciara->substates->Mb       = new (std::nothrow)   bool[sciara->domain->rows*sciara->domain->cols];
	sciara->substates->Msl      = new (std::nothrow) double[sciara->domain->rows*sciara->domain->cols];
}

void deallocateSubstates(Sciara *sciara)
{
	if(sciara->substates->Sz)       delete[] sciara->substates->Sz;
  if(sciara->substates->Sz_next)  delete[] sciara->substates->Sz_next;
	if(sciara->substates->Slt)      delete[] sciara->substates->Slt;
  if(sciara->substates->Slt_next) delete[] sciara->substates->Slt_next;
	if(sciara->substates->St)       delete[] sciara->substates->St;
  if(sciara->substates->St_next)  delete[] sciara->substates->St_next;
	if(sciara->substates->Sf)       delete[] sciara->substates->Sf;
//if(sciara->substates->Mv)       delete[] sciara->substates->Mv;
	if(sciara->substates->Mb)       delete[] sciara->substates->Mb;
	if(sciara->substates->Msl)      delete[] sciara->substates->Msl;
}

void init(Sciara*& sciara)
{
  sciara = new Sciara;
  sciara->domain = new Domain;
  sciara->substates = new Substates;
  //allocateSubstates(sciara); //Substates allocation is done when the confiugration is loaded
  sciara->parameters = new Parameters;
  sciara->simulation = new Simulation;
}

void finalize(Sciara*& sciara)
{
  deallocateSubstates(sciara);
  delete sciara->domain;
  delete sciara->substates;
  delete sciara->parameters;
  delete sciara->simulation;
  delete sciara;
  sciara = NULL;
}

int Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; // Xj: Moore neighborhood row coordinates (see below)
int Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; // Xj: Moore neighborhood col coordinates (see below)

void MakeBorder(Sciara *sciara) 
{
	int j, i;

	//prima riga
	i = 0;
	for (j = 0; j < sciara->domain->cols; j++)
		if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
			calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

	//ultima riga
	i = sciara->domain->rows - 1;
	for (j = 0; j < sciara->domain->cols; j++)
		if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
			calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

	//prima colonna
	j = 0;
	for (i = 0; i < sciara->domain->rows; i++)
		if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
			calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);
  
	//ultima colonna
	j = sciara->domain->cols - 1;
	for (i = 0; i < sciara->domain->rows; i++)
		if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
			calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);
	
	//il resto
	for (int i = 1; i < sciara->domain->rows - 1; i++)
		for (int j = 1; j < sciara->domain->cols - 1; j++)
			if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0) {
				for (int k = 1; k < MOORE_NEIGHBORS; k++)
					if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i+Xi[k], j+Xj[k]) < 0)
          {
			      calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);
						break;
					}
			}

}
