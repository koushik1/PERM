#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "jemalloc/jemalloc.h"

#define BACK_FILE "/tmp/app.back" /* Note: different backup and mmap files */
#define MMAP_FILE "/tmp/app.mmap"
#define MMAP_SIZE ((size_t)1 << 30)

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

int npoints = 128;
int npebs = 8;
double end_time = 1;
int nthreads = 4;
int narea = 128*128;

PERM double u_i0[npoints*npoints];
PERM double u_i1[npoints*npoints];
PERM double u_cpu[npoints*npoints];
PERM double pebs[npoints*npoints];
PERM double t;

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);

int main(int argc, char *argv[])
    {

    perm(PERM_START, PERM_SIZE);
    mopen(MMAP_FILE, mode, MMAP_SIZE);
    bopen(BACK_FILE, mode);

    double h;

    printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

    h = (XMAX - XMIN)/npoints;

    if (!do_restore) {
        t = 0.; 
        init_pebbles(pebs, npebs, npoints);
        init(u_i0, pebs, npoints);
        init(u_i1, pebs, npoints);
        print_heatmap("lake_i.dat", u_i0, npoints, h);
        mflush(); /* a flush is needed to save some global state */
        backup();
    }

    else {
        printf("restarting...\n");
        restore();
    }

    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
    print_heatmap("lake_f.dat", u_cpu, npoints, h);

    // Cleanup
    mclose();
    bclose();
    remove(BACK_FILE);
    remove(MMAP_FILE);
    return 1;
    }

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
    {
    double *un, *uc, *uo;
    double t, dt;

    un = (double*)malloc(sizeof(double) * n * n);
    uc = (double*)malloc(sizeof(double) * n * n);
    uo = (double*)malloc(sizeof(double) * n * n);

    memcpy(uo, u0, sizeof(double) * n * n);
    memcpy(uc, u1, sizeof(double) * n * n);

    t = 0.;
    dt = h / 2.;

    while(1)
    {
        evolve13pt(un, uc, uo, pebbles, n, h, dt, t);

        memcpy(uo, uc, sizeof(double) * n * n);
        memcpy(uc, un, sizeof(double) * n * n);

        if(!tpdt(&t,dt,end_time)) break;
        backup();
    }

    memcpy(u, un, sizeof(double) * n * n);
    }

void init_pebbles(double *p, int pn, int n)
    {
    int i, j, k, idx;
    int sz;

    srand( time(NULL) );
    memset(p, 0, sizeof(double) * n * n);

    for( k = 0; k < pn ; k++ )
    {
        i = rand() % (n - 4) + 2;
        j = rand() % (n - 4) + 2;
        sz = rand() % MAX_PSZ;
        idx = j + i * n;
        p[idx] = (double) sz;
    }
    }

double f(double p, double t)
    {
    return -expf(-TSCALE * t) * p;
    }

int tpdt(double *t, double dt, double tf)
    {
    if((*t) + dt > tf) return 0;
    (*t) = (*t) + dt;
    return 1;
    }

void init(double *u, double *pebbles, int n)
    {
    int i, j, idx;

    for(i = 0; i < n ; i++)
    {
        for(j = 0; j < n ; j++)
        {
        idx = j + i * n;
        u[idx] = f(pebbles[idx], 0.0);
        }
    }
    }

void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
    {
    int i, j, idx;

    for( i = 0; i < n; i++)
    {
        for( j = 0; j < n; j++)
        {
        idx = j + i * n;

        if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
        {
            un[idx] = 0.;
        }
        else
        {
            un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] +
                        uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));
        }
        }
    }
    }

void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
    {
    int i, j, idx;

    for( i = 0; i < n; i++)
    {
        for( j = 0; j < n; j++)
        {
        idx = j + i * n;

        if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
        {
            un[idx] = 0.;
        }
        else
        {
            un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *
                    ((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx-n] + // west east north south
                    0.25 * (uc[idx - n - 1] + uc[idx - n + 1] + uc[idx + n - 1] + uc[idx + n + 1]) + // northwest northeast southwest southeast
                    0.125 * (uc[idx - 2] + uc[idx + 2] + uc[idx - (2*n)] + uc[idx + (2*n)]) - // westwest easteast northnorth southsouth
                    5.5 * uc[idx])/(h * h) + f(pebbles[idx], t));
        }
        }
    }
    }

void print_heatmap(const char *filename, double *u, int n, double h)
    {
    int i, j, idx;

    FILE *fp = fopen(filename, "w");

    for( i = 0; i < n; i++ )
    {
        for( j = 0; j < n; j++ )
        {
        idx = j + i * n;
        fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
        }
    }

    fclose(fp);
    }
