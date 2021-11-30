#include "Sciara.h"

// ----------------------------------------------------------------------------
// Memory allocation function for 2D linearized buffers
// ----------------------------------------------------------------------------
void allocateSubstates(Sciara *sciara)
{
	sciara->substates->Sz       = new (std::nothrow) double[sciara->rows*sciara->cols];
  sciara->substates->Sz_next  = new (std::nothrow) double[sciara->rows*sciara->cols];
	sciara->substates->Slt      = new (std::nothrow) double[sciara->rows*sciara->cols];
  sciara->substates->Slt_next = new (std::nothrow) double[sciara->rows*sciara->cols];
	sciara->substates->St       = new (std::nothrow) double[sciara->rows*sciara->cols];
  sciara->substates->St_next  = new (std::nothrow) double[sciara->rows*sciara->cols];
	sciara->substates->Sf       = new (std::nothrow) double[sciara->rows*sciara->cols*NUMBER_OF_OUTFLOWS];
	sciara->substates->Mv       = new (std::nothrow) int[sciara->rows*sciara->cols];
	sciara->substates->Mb       = new (std::nothrow) bool[sciara->rows*sciara->cols];
	sciara->substates->Msl      = new (std::nothrow) double[sciara->rows*sciara->cols];
}

void deallocateSubstates(Sciara *sciara)
{
	delete[] sciara->substates->Sz;
  delete[] sciara->substates->Sz_next;
	delete[] sciara->substates->Slt;
  delete[] sciara->substates->Slt_next;
	delete[] sciara->substates->St;
  delete[] sciara->substates->St;
	delete[] sciara->substates->Sf;
	delete[] sciara->substates->Mv;
	delete[] sciara->substates->Mb;
	delete[] sciara->substates->Msl;
}
