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

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mf
{
	enum {
		P_L2_MFR = 0,
		P_L1_MFR = 1,
		P_KL_MFR = 2,
		P_LR_MFC = 5,
		P_L2_MFC = 6,
		P_L1_MFC = 7,
		P_ROW_BPR_MFOC = 10,
		P_COL_BPR_MFOC = 11
	};

	enum {
		RMSE = 0,
		MAE = 1,
		GKL = 2,
		LOGLOSS = 5,
		ACC = 6,
		ROW_MPR = 10,
		COL_MPR = 11,
		ROW_AUC = 12,
		COL_AUC = 13
	};

	typedef float mf_float;
	typedef double mf_double;
	typedef long long mf_int;
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
		short *P;
		short *Q;
	};

	struct mf_problem
	{
		mf_int m;
		mf_int n;
		mf_long nnz;
		struct mf_node *R;
	};

	struct blocks 
	{
		struct mf_node *block1;
		struct mf_node *block2;
		struct mf_node *block3;
		struct mf_node *block4;
	}



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

	mf_problem read_problem(string path);
	blocks partition(mf_problem prob);
	mf_model* mf_load_model(string path);

}
#endif // _MF_GPU_H