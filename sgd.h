#ifndef _SGD_GPU_H
#define _SGD_GPU_H

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
#include <curand.h>
#include <curand_kernel.h>
#include "../common/common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mf.h"
#include "sgd_kernel.h"

extern __global__ void init_rand_state(curandState*state, int size);

__global__ void init_rand_state(curandState* state, int size) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < size) {
		curand_init(clock() + tid, tid, 0, &state[tid]);
	}
}

using namespace std;

namespace sgd
{
	/** @brief performs Stochastic Gradient Decsent training on the dataset and returns a trained
	*			model
	* @param char const *path: path to file later
	* @param mf_problem* tr: problem to train on
	* @param hog_params params: hogwild parameters
	*/
	void sgd_train(mf_problem* tr, mf_model* model)
	{
		// Load Params
		hog_params params;
		float dynamic_rate[1024];
		float alpha = params.alpha;
		float beta = params.beta;
		double lambda_p = params.lambda_p;
		double lambda_q = params.lambda_q;
		int num_iters = params.max_iters;
		
		/////////////////////////////
		//		Set up Cuda Dev    //
		/////////////////////////////
		int dev = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		printf("at Device %d: %s\n", dev, deviceProp.name);
		cudaSetDevice(dev);

		//////////////////////////////
		//			Calc LR			//
		//////////////////////////////
		printf("\n\tCalculating Learning Rates\n");
		for (int i = 0; i < (params.max_iters + 4); i++) {
			float tmp_rate = alpha / (1 + beta*pow(i, 1.5));
			dynamic_rate[i] = tmp_rate;
		}
		// Initialize Device Variables
		printf("\n\tInitializing Device Variables\n");
		float *gpu_dynamic_rate;
		cudaMalloc((float**)&gpu_dynamic_rate, sizeof(float) * 1024);
		cudaMemcpy(gpu_dynamic_rate, dynamic_rate, sizeof(float) * 1024, cudaMemcpyHostToDevice);

		//////////////////////////////
		//		Init Rand St		//
		//////////////////////////////
		curandState *rand_state;
		cudaMalloc(&rand_state, sizeof(curandState)*params.num_warps);
		init_rand_state << <(params.num_warps+255)/256, 256 >> > (rand_state, params.num_warps);

		//////////////////////////////
		//		Set up GPU Vars		//
		//////////////////////////////
		cudaMalloc(&(tr->d_R), sizeof(mf_node)*tr->max_count);
		tr->u_id = -1;
		tr->v_id = -1;

		cudaMalloc(&(model->d_P), sizeof(half)*model->u_seg_len*model->k);
		cudaMalloc(&(model->d_Q), sizeof(half)*model->v_seg_len*model->k);
		model->u_id = -1;
		model->v_id = -1;


		//////////////////////////////
		//	  Set Up Train Params	//
		//////////////////////////////
		int update_vector_size = params.batch_size;
		int *update_count_per_partition = new int[params.x_part*params.y_part];
		long long max_updates = 0;
		for (int i = 0; i < params.x_part * params.y_part; i++) {
			long long cur_size = (ceil)(1.0*(tr->counts[i]) / (update_vector_size * params.num_warps * 10) );
			update_count_per_partition[i] = cur_size;
			if (max_updates < cur_size) {
				max_updates = cur_size;
			}
		}
		vector<int> u_ids(params.x_part, 0);
		vector<int> v_ids(params.y_part, 0);
		for (int i = 0; i < params.y_part; i++) {
			u_ids[i] = i;
		}
		for (int i = 0; i < params.x_part; i++) {
			v_ids[i] = i;
		}


		for (int cur_iter = 0; cur_iter < params.max_iters; cur_iter++) {
			printf("Progress:\t%d of %d\n", cur_iter, num_iters);
			printf("Working on Parts: ");
			for (int u_sec = 0; u_sec < params.y_part; u_sec++) {
				for (int v_sec = 0; v_sec < params.x_part; v_sec++) {
					int u_id = u_ids[u_sec];
					int v_id = v_ids[v_sec];

					int current_partition = u_id*params.y_part + v_id;
					printf("%d\t", current_partition);
					if (tr->u_id != u_id || tr->v_id != v_id) {
						// Copy over particular data of (partition size) starting from
						// beginning of partition
						cudaMemcpy(tr->d_R, tr->R_ptrs[current_partition], sizeof(mf_node)*tr->counts[current_partition], cudaMemcpyHostToDevice);
					}
					// Set Markers for current partition
					tr->u_id = u_id;
					tr->v_id = v_id;
					// Transfer P to GPU
					if (model->u_id == -1) {
						// First Iteration
						short *p_tmp = model->P + (long long)(model->u_seg_len*model->k*u_id);
						cudaMemcpy(model->d_P, p_tmp, sizeof(half)*model->u_seg_len*model->k, cudaMemcpyHostToDevice);
						//cudaDeviceSynchronize();
					}
					else if (model->u_id != u_id) {
						short *p_tmp = model->P + model->u_seg_len*model->k*model->u_id;
						cudaMemcpy(p_tmp, model->d_P, sizeof(half)*model->u_seg_len*model->k, cudaMemcpyHostToDevice);
						//cudaDeviceSynchronize();

						p_tmp = model->P + (long long)(model->u_seg_len*model->k*u_id);
						cudaMemcpy(model->d_P, p_tmp, sizeof(half)*model->u_seg_len*model->k, cudaMemcpyHostToDevice);
						//cudaDeviceSynchronize();
					}
					model->u_id = u_id;
					// Transfer Q to GPU
					if (model->v_id == -1) {
						// First Iteration
						short *q_tmp = model->Q + (long long)(model->v_seg_len*model->k*v_id);
						cudaMemcpy(model->d_Q, q_tmp, sizeof(half)*model->v_seg_len*model->k, cudaMemcpyHostToDevice);
						//cudaDeviceSynchronize();
						
					}
					else if (model->v_id != v_id) {
						short *q_tmp = model->Q + (long long)(model->v_seg_len*model->k*model->v_id);
						cudaMemcpy(q_tmp, model->d_Q, sizeof(half)*model->v_seg_len*model->k, cudaMemcpyDeviceToHost);
						cudaDeviceSynchronize();

						q_tmp = model->Q + model->v_seg_len*model->k*v_id;
						cudaMemcpy(model->d_Q, q_tmp, sizeof(half)*model->v_seg_len*model->k, cudaMemcpyHostToDevice);
						cudaDeviceSynchronize();

					}
					model->v_id = v_id;
					hogwild_train << <params.num_warps / 4, 128 >> > (
						tr->d_R,
						tr->counts[current_partition],
						model->k,
						model->d_P,
						model->d_Q,
						rand_state,
						gpu_dynamic_rate,
						cur_iter,
						1,
						lambda_p,
						lambda_q,
						update_count_per_partition[current_partition]);
					cudaDeviceSynchronize();
				}
			}
			printf("\n\n");
		}
		cudaDeviceSynchronize();
		if (model->u_id >= 0) {
			short *p_tmp = model->P + (long long)(model->u_seg_len*model->k*model->u_id);
			cudaMemcpy(p_tmp, model->d_P, sizeof(half)*model->u_seg_len*model->k, cudaMemcpyDeviceToHost);
		}
		if (model->v_id >= 0) {
			short *q_tmp = model->Q + (long long)(model->v_seg_len*model->k*model->v_id);
			cudaMemcpy(q_tmp, model->d_Q, sizeof(half)*model->v_seg_len*model->k, cudaMemcpyDeviceToHost);
		}

		cudaFree(tr->d_R);
		cudaFree(model->d_P);
		cudaFree(model->d_Q);
		cudaFree(gpu_dynamic_rate);
		cudaFree(rand_state);
		cudaDeviceSynchronize();
	}
}
#endif //_SGD_GPU_H