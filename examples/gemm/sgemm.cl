/*
Original works by:
--------------------------------------------------------
MAGMA
Copyright (c) 2017 The University of Tennessee. All rights reserved.
Licensed under modified BSD license
*/


// These parameters will be determined by utils.read_code
//#define DIM_X  ${DIM_X}
//#define DIM_Y  ${DIM_Y}
//#define BLK_M  ${BLK_M}
//#define BLK_N  ${BLK_N}
//#define BLK_K  ${BLK_K}
//#define DIM_XA  ${DIM_XA}
//#define DIM_YA  ${DIM_YA}
//#define DIM_XB  ${DIM_XB}
//#define DIM_YB  ${DIM_YB}
//#define THR_N  ${THR_N}
//#define THR_M  ${THR_M}

#include<clpy/carray.clh>

#define fetch(arr, offs, col, m, n, bound) arr[offs + (long)min((long)((n)*(col) + m), (long)bound)]


__kernel void sgemm(
        long M, long N, long K, // np.int_ in Python
        __global const float* A,
        CArray_2 info_A,
        __global const float* B,
        CArray_2 info_B,
        __global float * C,
        CArray_2 info_C
) {
    int idx = get_local_id(0);
    int idy = get_local_id(1);

    int idt = DIM_X * idy + idx;

    int idxA = idt % DIM_XA;
    int idyA = idt / DIM_XA;

    int idxB = idt % DIM_XB;
    int idyB = idt / DIM_XB;

    int blx = get_group_id(0);
    int bly = get_group_id(1);

    __local float sA[BLK_K][BLK_M + 1];
    __local float sB[BLK_N][BLK_K + 1];

    // registers for the innermost loop
    float rC[THR_N][THR_M];
    float rA[THR_M];
    float rB[THR_N];

    float ra[BLK_K / DIM_YA][BLK_M / DIM_XA];
    float rb[BLK_N / DIM_YB][BLK_K / DIM_XB];

    int offs_dA = blx * BLK_M     + idyA * M + idxA;
    int boundA = (M * (K - 1) + M) - (blx * BLK_M + idyA * M + idxA) - 1;
    int offs_dB = bly * BLK_N * K + idyB * K + idxB;
    int boundB = (K * (N - 1) + K) - (bly * BLK_N * K + idyB * K + idxB) - 1;

    int m, n, k, kk;
    
    for (n = 0; n < THR_N; n++) {
        for (m = 0 ; m < THR_M; m++) {
            rC[n][m] = 0;
        }
    }

    // blockwise transpose to transpose load
    for (n = 0; n < BLK_K; n += DIM_YA) {
        for (m = 0; m < BLK_M; m += DIM_XA) {
            sA[n + idyA][m + idxA] = fetch(A, offs_dA, M, m, n, boundA);
        }
    }
    // blockwise transpose to transpose load
    for (n = 0; n < BLK_N; n += DIM_YB) {
        for (m = 0; m < BLK_K; m += DIM_XB) {
            sB[n + idyB][m + idxB] = fetch(B, offs_dB, K, m, n, boundB);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (kk = 0; kk < K - BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K * M;
        boundA -= BLK_K * M;
        offs_dB += BLK_K;
        boundB -= BLK_K;
        
        for (n = 0; n < BLK_K / DIM_YA; n++) {
            for (m = 0; m < BLK_M / DIM_XA; m++) {
                ra[n][m] = fetch(A, offs_dA, M, m * DIM_XA, n * DIM_YA, boundA);
            }
        }

        for (n = 0; n < BLK_N / DIM_YB; n++) {
            for (m = 0; m < BLK_K / DIM_XB; m++) {
                rb[n][m] = fetch(B, offs_dB, K, m * DIM_XB, n * DIM_YB, boundB);
            }
        }

        // multiply
        for (k = 0; k < BLK_K; k++)
        {
            for (m = 0; m < THR_M; m++) {
                rA[m] = sA[k][m * DIM_X + idx];
            }
            
            for (n = 0; n < THR_N; n++) {
                rB[n] = sB[n * DIM_Y + idy][k];
            }

            for (n = 0; n < THR_N; n++) {  
                for (m = 0; m < THR_M; m++) {
                    rC[n][m] += rA[m] * rB[n];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // store A regs->smem
        
        for (n = 0; n < BLK_K / DIM_YA; n++)
        {
            for (m = 0; m < BLK_M / DIM_XA; m++)
            {
                sA[n * DIM_YA + idyA][m * DIM_XA + idxA] = ra[n][m];
            }
        }

        
        for (n = 0; n < BLK_N / DIM_YB; n++)
        {
            for (m = 0; m < BLK_K / DIM_XB; m++)
            {
                sB[n * DIM_YB + idyB][m * DIM_XB + idxB] = rb[n][m];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Multiply last full (BLK_K) or partial block of columns of A and
    // rows of B.
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.

    kk = K - kk;
    for (k = 0; k < kk; k++)
    {
        for (m = 0; m < THR_M; m++) {
            rA[m] = sA[k][m * DIM_X + idx];
        }

        for (n = 0; n < THR_N; n++) {
            rB[n] = sB[n * DIM_Y + idy][k];
        }
        
        for (n = 0; n < THR_N; n++) {
            for (m = 0; m < THR_M; m++) {
                rC[n][m] += rA[m] * rB[n];
            }
        }
    }

    
    for (n = 0; n < THR_N; n++) {
        int coord_dCn = bly * BLK_N + n * DIM_Y + idy;
        for (m = 0; m < THR_M; m++) {
            int coord_dCm = blx * BLK_M + m * DIM_X + idx;
            if (coord_dCm < M && coord_dCn < N) {
                C[coord_dCn * M + coord_dCm] = rC[n][m];
            }
        }
    }
}
