#include "Sciara.h"
#include "io.h"
#include "util.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using std::vector;

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
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
static inline void emitLava(
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
    double *Sh,
    double *Sh_next,
    double *ST_next,
    double &thread_local_emitted)
{
    for (int k = 0; k < (int)vent.size(); k++)
    {
        if (i == vent[k].y() && j == vent[k].x())
        {
            double th = vent[k].thickness(elapsed_time, Pclock, emission_time, Pac);
            SET(Sh_next, c, i, j, GET(Sh, c, i, j) + th);
            SET(ST_next, c, i, j, PTvent);
            thread_local_emitted += th;
        }
    }
}

static inline void computeOutflows(
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
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];
    bool loop;
    int counter;
    double sz0, sz, T, avg, rr, hc;

    if (GET(Sh, c, i, j) <= 0.0)
        return;

    T  = GET(ST, c, i, j);
    rr = pow(10.0, _a + _b * T);
    hc = pow(10.0, _c + _d * T);

    for (int k = 0; k < MOORE_NEIGHBORS; k++)
    {
        sz0 = GET(Sz, c, i, j);
        sz  = GET(Sz, c, i + Xi[k], j + Xj[k]);
        h[k]  = GET(Sh, c, i + Xi[k], j + Xj[k]);
        w[k]  = Pc;
        Pr[k] = rr;

        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz;
        else
            z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
    }

    H[0] = z[0];
    theta[0] = 0.0;
    eliminated[0] = false;

    for (int k = 1; k < MOORE_NEIGHBORS; k++)
    {
        if (z[0] + h[0] > z[k] + h[k])
        {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
            eliminated[k] = false;
        }
        else
        {
            eliminated[k] = true;
        }
    }

    do
    {
        loop = false;
        avg = h[0];
        counter = 0;
        for (int k = 0; k < MOORE_NEIGHBORS; k++)
        {
            if (!eliminated[k])
            {
                avg += H[k];
                counter++;
            }
        }
        if (counter != 0)
            avg = avg / (double)counter;

        for (int k = 0; k < MOORE_NEIGHBORS; k++)
        {
            if (!eliminated[k] && avg <= H[k])
            {
                eliminated[k] = true;
                loop = true;
            }
        }
    } while (loop);

    for (int k = 1; k < MOORE_NEIGHBORS; k++)
    {
        if (!eliminated[k] && h[0] > hc * cos(theta[k]))
            BUF_SET(Mf, r, c, k - 1, i, j, Pr[k] * (avg - H[k]));
        else
            BUF_SET(Mf, r, c, k - 1, i, j, 0.0);
    }
}

static inline void massBalance(
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

    double inFlow, outFlow, neigh_t;
    double initial_h = GET(Sh, c, i, j);
    double initial_t = GET(ST, c, i, j);
    double h_next = initial_h;
    double t_next = initial_h * initial_t;

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
    else
    {
        // opzionale: azzera se serve
        SET(ST_next, c, i, j, 0.0);
        SET(Sh_next, c, i, j, 0.0);
    }
}

static inline void computeNewTemperatureAndSolidification(
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
    (void)r; (void)c; (void)Mf; // non usati qui, ma lasciati per coerenza

    double z = GET(Sz, c, i, j);
    double h = GET(Sh, c, i, j);
    double T = GET(ST, c, i, j);

    if (h > 0.0 && GET(Mb, c, i, j) == false)
    {
        double aus = 1.0 + (3.0 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) /
                            (Prho * Pcv * h * Pac);
        double nT = T / pow(aus, 1.0 / 3.0);

        if (nT > PTsol)
        {
            SET(ST_next, c, i, j, nT);
            // Sh_next normalmente già copiato altrove; qui non lo tocchiamo
        }
        else
        {
            SET(Sz_next, c, i, j, z + h);
            SET(Sh_next, c, i, j, 0.0);
            SET(ST_next, c, i, j, PTsol);
            SET(Mhs, c, i, j, GET(Mhs, c, i, j) + h);
        }
    }
}

static inline void boundaryConditions(
    int i,
    int j,
    int r,
    int c,
    double *Mf,
    bool *Mb,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next)
{
    (void)Mf; (void)Sh; (void)ST;

    // Imposta a zero le celle marcate come boundary (Mb==true)
    if (GET(Mb, c, i, j))
    {
        SET(Sh_next, c, i, j, 0.0);
        SET(ST_next, c, i, j, 0.0);
    }
}

static inline double reduceAdd(int r, int c, double *buffer)
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
    if (argc < 6)
    {
        fprintf(stderr,
                "Usage: %s input.cfg output.cfg max_steps reduce_interval thickness_threshold\n",
                argv[0]);
        return 1;
    }

    Sciara *sciara = new Sciara();
    init(sciara);

    int max_steps = atoi(argv[MAX_STEPS_ID]);
    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    int r = sciara->domain->rows;
    int c = sciara->domain->cols;

    if (r <= 2 || c <= 2)
    {
        fprintf(stderr, "Error: domain too small (rows=%d cols=%d)\n", r, c);
        finalize(sciara);
        return 2;
    }

    // Range completo (per kernel che NON accedono ai vicini)
    int i_start = 0, i_end = r;
    int j_start = 0, j_end = c;

    // Range interno (per kernel che accedono al vicinato: evita OOB)
    int i_nb_start = 1, i_nb_end = r - 1;
    int j_nb_start = 1, j_nb_end = c - 1;

    double total_current_lava = -1.0;

    simulationInitialize(sciara);

    util::Timer cl_timer;

    while ((max_steps > 0 && sciara->simulation->step < max_steps) ||
           (sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
           (total_current_lava < 0.0 || total_current_lava > thickness_threshold))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        // ---------------- emitLava (tutto il dominio, non legge vicini) ----------------
        double emitted_step_total = 0.0;

        #pragma omp parallel
        {
            double emitted_local = 0.0;

            #pragma omp for
            for (int i = i_start; i < i_end; i++)
            {
                for (int j = j_start; j < j_end; j++)
                {
                    emitLava(i, j, r, c,
                             sciara->simulation->vent,
                             sciara->simulation->elapsed_time,
                             sciara->parameters->Pclock,
                             sciara->simulation->emission_time,
                             sciara->parameters->Pac,
                             sciara->parameters->PTvent,
                             sciara->substates->Sh,
                             sciara->substates->Sh_next,
                             sciara->substates->ST_next,
                             emitted_local);
                }
            }

            #pragma omp atomic
            emitted_step_total += emitted_local;
        }

        sciara->simulation->total_emitted_lava += emitted_step_total;

        memcpy(sciara->substates->Sh, sciara->substates->Sh_next, sizeof(double) * r * c);
        memcpy(sciara->substates->ST, sciara->substates->ST_next, sizeof(double) * r * c);

        // ---------------- computeOutflows (SOLO interno: legge vicini) ----------------
        #pragma omp parallel for
        for (int i = i_nb_start; i < i_nb_end; i++)
        {
            for (int j = j_nb_start; j < j_nb_end; j++)
            {
                computeOutflows(i, j, r, c,
                                sciara->X->Xi, sciara->X->Xj,
                                sciara->substates->Sz,
                                sciara->substates->Sh,
                                sciara->substates->ST,
                                sciara->substates->Mf,
                                sciara->parameters->Pc,
                                sciara->parameters->a,
                                sciara->parameters->b,
                                sciara->parameters->c,
                                sciara->parameters->d);
            }
        }

        // ---------------- massBalance (SOLO interno: legge vicini) ----------------
        #pragma omp parallel for
        for (int i = i_nb_start; i < i_nb_end; i++)
        {
            for (int j = j_nb_start; j < j_nb_end; j++)
            {
                massBalance(i, j, r, c,
                            sciara->X->Xi, sciara->X->Xj,
                            sciara->substates->Sh,
                            sciara->substates->Sh_next,
                            sciara->substates->ST,
                            sciara->substates->ST_next,
                            sciara->substates->Mf);
            }
        }

        memcpy(sciara->substates->Sh, sciara->substates->Sh_next, sizeof(double) * r * c);
        memcpy(sciara->substates->ST, sciara->substates->ST_next, sizeof(double) * r * c);

        // ---------------- computeNewTemperatureAndSolidification (tutto il dominio) ----------------
        #pragma omp parallel for
        for (int i = i_start; i < i_end; i++)
        {
            for (int j = j_start; j < j_end; j++)
            {
                computeNewTemperatureAndSolidification(i, j, r, c,
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
            }
        }

        memcpy(sciara->substates->Sz, sciara->substates->Sz_next, sizeof(double) * r * c);
        memcpy(sciara->substates->Sh, sciara->substates->Sh_next, sizeof(double) * r * c);
        memcpy(sciara->substates->ST, sciara->substates->ST_next, sizeof(double) * r * c);

        // ---------------- boundaryConditions (tutto il dominio) ----------------
        #pragma omp parallel for
        for (int i = i_start; i < i_end; i++)
        {
            for (int j = j_start; j < j_end; j++)
            {
                boundaryConditions(i, j, r, c,
                                   sciara->substates->Mf,
                                   sciara->substates->Mb,
                                   sciara->substates->Sh,
                                   sciara->substates->Sh_next,
                                   sciara->substates->ST,
                                   sciara->substates->ST_next);
            }
        }

        memcpy(sciara->substates->Sh, sciara->substates->Sh_next, sizeof(double) * r * c);
        memcpy(sciara->substates->ST, sciara->substates->ST_next, sizeof(double) * r * c);

        // ---------------- Global reduction ----------------
        if (reduceInterval > 0 && (sciara->simulation->step % reduceInterval == 0))
            total_current_lava = reduceAdd(r, c, sciara->substates->Sh);
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
