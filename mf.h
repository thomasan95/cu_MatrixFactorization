#ifndef _MF_GPU_H
#define _MF_GPU_H
#include "../common/common.h"
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

#include "../half-1.12.0/half.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
using namespace std;

struct mf_node
{
	long long u;
	long long v;
	float r;
};

struct mf_model
{
	long long fun;
	long long m;
	long long n;
	long long k;
	float b;

	int u_grid, v_grid;
	int x_grid, y_grid;
	int ux, uy;
	long long u_seg_len, v_seg_len;
	short *P;
	short *Q;

	half* d_P;
	half* d_Q;

	int u_id, v_id;
};

struct mf_problem
{
	long long m;
	long long n;
	long long nnz;

	int x_part;
	int y_part;

	long long *counts;

	struct mf_node *R;
	struct mf_node **R_ptrs;

	struct mf_node *d_R;

	long long max_count;

	int u_id, v_id;

	long long u_seg_len, v_seg_len;
};
template <typename T> T* malloc_aligned_float(long long size)
{
	long long const kALIGNByte = 32;
	long long const kALIGN = kALIGNByte / sizeof(T);

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
struct hog_params
{
	int x_part = 2;
	int y_part = 2;
	int batch_size = 128;
	int max_iters = 30;
	double lambda_p = 0.0461;
	double lambda_q = 0.0451;
	double alpha = 0.05;
	double beta = 0.3;
	int num_warps = 750;
};

void partition(mf_problem* prob, int x_part, int y_part);
mf_problem read_problem(string path);
mf_model* mf_init_model(const char* path, mf_problem& prob);
void mf_destroy_model(mf_model ** model);

#endif // _MF_GPU_H