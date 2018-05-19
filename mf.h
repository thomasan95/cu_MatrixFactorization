#ifndef _MF_GPU_H
#define _MF_GPU_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../common/common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
using namespace std;

typedef float mf_float; // SGDRate
typedef double mf_double;
typedef long long mf_int; // SGDIndex
typedef long long mf_long;

struct mf_node
{
	mf_int u;
	mf_int v;
	mf_float r;
};

struct mf_model
{
	mf_int fun;
	mf_int m;
	mf_int n;
	mf_int k;
	mf_float b;


	int u_grid, v_grid;
	int x_grid, y_grid;
	int ux, uy;
	long long u_seg, v_seg;
	short *P;
	short *Q;

	half *gpuHalfP;
	half *gpuHalfQ;

	int cur_u_id;
	int cur_v_id;
	half *gpuHalfPptrs[2];
	half *gpuHalfQptrs[2];

	int cur_global_x_id[2];
	int cur_global_y_id[2];
	

};

struct mf_problem
{
	mf_int m;
	mf_int n;
	mf_long nnz;
	mf_long count1;
	mf_long count2;
	mf_long count3;
	mf_long count4;
	struct mf_node *R;
	struct mf_node **R2D;
	struct mf_node *gpuR;

	struct mf_node *block1;
	struct mf_node *block2;
	struct mf_node *block3;
	struct mf_node *block4;

	struct mf_node *gpuB1;
	struct mf_node *gpuB2;
	struct mf_node *gpuB3;
	struct mf_node *gpuB4;

	int x_part = 2;
	int y_part = 2;

	long long* gridSize;
	long long maxGridSize;

	mf_long u_seg_len, v_seg_len;

	int cur_u_id;
	int cur_v_id;

	struct mf_node *gpuRptrs[2];
	int cur_global_xid[2];
	int cur_global_yid[2];
};

struct hog_params
{
	int x_part = 2;
	int y_part = 2;
	int batch_size = 128;
	int max_iters = 20;
	double lambda_p = 0.0461;
	double lambda_q = 0.0451;
	double alpha = 0.08;
	double beta = 0.3;
};

template <typename T> T* malloc_aligned_float(mf_long size)
{
	mf_int const kALIGNByte = 32;
	mf_int const kALIGN = kALIGNByte / sizeof(T);

	void *ptr;
#ifdef _WIN32
	ptr = _aligned_malloc(size * sizeof(T), kALIGNByte);
	if (ptr == nullptr)
		throw bad_alloc();
#else
	int status = posix_memalign(&ptr, kALIGNByte, size * sizeof(T));
	if (status != 0)
		throw bad_alloc();
#endif

	return (T*)ptr;
}
void destroy_blocks(mf_problem & b);
void partition(mf_problem& prob);
mf_problem read_problem(string path);
mf_model* mf_init_model(const char* path);
void mf_destroy_model(mf_model ** model);

#endif // _MF_GPU_H