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
#include "../common/common.h"
#include <curand.h>
#include <curand_kernel.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mf.h"

// Perform random updates
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
	__global__ void hogwild_train(const mf_node*R, 
								mf_long nnz,
								int k,
								short* d_P,
								short* d_Q,
								curandState *state,
								float* dynamic_rate,
								int cur_iter,
								int num_iters,
								double lambda_p,
								double lambda_q,
								int vec_size,
								int update_count_per_block,
								int update_count_this_block) 
	{	
		//////////////////////////////////////////////////////////
		//=====UNCOMMENT WHEN DONE DEBUGGING===============
		//for (int itr = cur_iter; itr < cur_iter + num_iters; itr++) {
		//////////////////////////////////////////////////////////
		for (int itr = 0; itr < 1; itr++) {
			float tmp_lr = __ldg(&dynamic_rate[itr]);
			//////////////////////////////////////////////////////////
			//=====UNCOMMENT WHEN DONE DEBUGGING===============
			//for (int update_itr = 0; update_itr < update_count_this_block; update_itr++) {
			//////////////////////////////////////////////////////////
			for (int update_itr = 0; update_itr < 1; update_itr++) {
				int lane_id = threadIdx.x % 32;
				int local_wid = threadIdx.x / 32;
				int wid = 4 * blockIdx.x + local_wid;
				mf_long start_id = 0;
				if (lane_id == 0) {
					mf_long origin = (mf_long)(curand_uniform(&state[wid])*nnz);
					start_id = origin % nnz;
				}
				// Set Start ID for All Threads
				start_id = __shfl(start_id, 0);
				for (int i = 0; i < vec_size; i++) {
					int offset = (start_id + i) % nnz;

					float r = __ldg(&R[offset].r);
					int u = __ldg(&R[offset].u);
					int v = __ldg(&R[offset].v);
					//Read P and Q into register
					int base_p = u*k;
					int base_q = v*k;

					float tmp_p1 = __half2float(d_P[base_p + lane_id]);
					float tmp_q1 = __half2float(d_Q[base_q + lane_id]);

					float tmp_p2 = __half2float(d_P[base_p + lane_id + 32]);
					float tmp_q2 = __half2float(d_Q[base_q + lane_id + 32]);

					float tmp_p3 = __half2float(d_P[base_p + lane_id + 64]);
					float tmp_q3 = __half2float(d_Q[base_q + lane_id + 64]);

					float tmp_p4 = __half2float(d_P[base_p + lane_id + 96]);
					float tmp_q4 = __half2float(d_Q[base_q + lane_id + 96]);

					float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;
					tmp_product += __shfl_down(tmp_product, 16);
					tmp_product += __shfl_down(tmp_product, 8);
					tmp_product += __shfl_down(tmp_product, 4);
					tmp_product += __shfl_down(tmp_product, 2);
					tmp_product += __shfl_down(tmp_product, 1);

					tmp_product = __shfl(tmp_product, 0);

					float r_uv = r - tmp_product;

					d_P[base_p + lane_id] = __float2half(tmp_p1 + tmp_lr*(r_uv*tmp_q1 - lambda_p*tmp_p4));
					d_Q[base_p + lane_id] = __float2half(tmp_q1 + tmp_lr*(r_uv*tmp_p1 - lambda_p*tmp_q4));

					d_P[base_p + lane_id + 32] = __float2half(tmp_p2 + tmp_lr*(r_uv*tmp_q2 - lambda_p*tmp_p4));
					d_Q[base_p + lane_id + 32] = __float2half(tmp_q2 + tmp_lr*(r_uv*tmp_p2 - lambda_p*tmp_q4));

					d_P[base_p + lane_id + 64] = __float2half(tmp_p3 + tmp_lr*(r_uv*tmp_q3 - lambda_p*tmp_p4));
					d_Q[base_p + lane_id + 64] = __float2half(tmp_q3 + tmp_lr*(r_uv*tmp_p3 - lambda_p*tmp_q4));

					d_P[base_p + lane_id + 96] = __float2half(tmp_p4 + tmp_lr*(r_uv*tmp_q4 - lambda_p*tmp_p4));
					d_Q[base_p + lane_id + 96] = __float2half(tmp_q4 + tmp_lr*(r_uv*tmp_p4 - lambda_p*tmp_q4));

				}
			}
		}
	}

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
		// Calculate Learning Rates Before Hand
		printf("\n\tCalculating Learning Rates\n");
		for (int i = 0; i < params.max_iters + 4; i++) {
			float tmp_rate = alpha / (1 + beta*pow(i, 1.5));
			dynamic_rate[i] = tmp_rate;
		}

		// Load CUDA
		int dev = 0;
		cudaDeviceProp deviceProp;
		CHECK(cudaGetDeviceProperties(&deviceProp, dev));
		printf("at Device %d: %s\n", dev, deviceProp.name);
		CHECK(cudaSetDevice(dev));

		// Initialize Random State
		curandState *rand_state;
		CHECK(cudaMalloc(&rand_state, sizeof(curandState)));
		init_rand_state << <1, 256 >> > (rand_state, 1);

		// Initialize Device Variables
		printf("\n\tInitializing Device Variables\n");
		short *d_P, *d_Q;
		float *gpu_dynamic_rate;
		CHECK(cudaMalloc((void**)&gpu_dynamic_rate, sizeof(float) * 1024));
		CHECK(cudaMemcpy(gpu_dynamic_rate, dynamic_rate, sizeof(float) * 1024, cudaMemcpyHostToDevice));
		CHECK(cudaMalloc((short**)&d_P, sizeof(short)*model->m*model->k));
		CHECK(cudaMalloc((short**)&d_Q, sizeof(short)*model->n*model->k));
		CHECK(cudaMemcpy(d_P, model->P, sizeof(short)*model->m*model->k, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Q, model->Q, sizeof(short)*model->n*model->k, cudaMemcpyHostToDevice));
		mf_node* d_R;

		// Define Iteration Specifics inside Kernel
		int update_vector_size = 128;
		int *update_count_per_partition = new int[params.x_part*params.y_part];
		update_count_per_partition[0] = (ceil)(1.0*(tr->count1) / update_vector_size);
		update_count_per_partition[1] = (ceil)(1.0*(tr->count2) / update_vector_size);
		update_count_per_partition[2] = (ceil)(1.0*(tr->count3) / update_vector_size);
		update_count_per_partition[3] = (ceil)(1.0*(tr->count4) / update_vector_size);
		int max_update_count_per_block = -1;
		for (int a = 0; a < params.x_part*params.y_part; a++) {
			if (max_update_count_per_block < update_count_per_partition[a]) {
				max_update_count_per_block = update_count_per_partition[a];
			}
		}
		printf("Starting batch_Hogwild SGD\n");
		for (int cur_iter = 0; cur_iter < 1; cur_iter++) {
			printf("\nProgress:\t\t%d/%d Iterations\n\n", cur_iter+1, num_iters);
			// Work on Specific Partition of the Data
			for (int part = 1; part <= 1; part++) {//params.x_part*params.y_part; part++) {
				printf("Working on Part %d\n", part);
				if (part == 1) {
					CHECK(cudaMalloc((void**)&d_R, sizeof(mf_node)*(tr->count1)));
					CHECK(cudaMemcpy(d_R, tr->block1, sizeof(mf_node)*(tr->count1), cudaMemcpyHostToDevice));
					printf("Calling HogWild!!!\n");
					sgd::hogwild_train << <1, 128 >> > (
						d_R,
						tr->nnz,
						model->k,
						d_P,
						d_Q,
						rand_state,
						gpu_dynamic_rate,
						cur_iter,
						num_iters,
						lambda_p,
						lambda_q,
						update_vector_size,
						max_update_count_per_block,
						update_count_per_partition[0]);
				}
				else if (part == 2) {
					CHECK(cudaMalloc((void**)&d_R, sizeof(mf_node)*tr->count2));
					CHECK(cudaMemcpy(d_R, tr->block1, sizeof(mf_node)*(tr->count2), cudaMemcpyHostToDevice));
					sgd::hogwild_train << <1, 128 >> > (
						d_R,
						tr->nnz,
						model->k,
						d_P,
						d_Q,
						rand_state,
						gpu_dynamic_rate,
						cur_iter,
						num_iters,
						lambda_p,
						lambda_q,
						update_vector_size,
						max_update_count_per_block,
						update_count_per_partition[1]);
				}
				else if (part == 3) {
					CHECK(cudaMalloc((void**)&d_R, sizeof(mf_node)*tr->count3));
					CHECK(cudaMemcpy(d_R, tr->block1, sizeof(mf_node)*(tr->count3), cudaMemcpyHostToDevice));
					sgd::hogwild_train << <1, 128 >> > (
						d_R,
						tr->nnz,
						model->k,
						d_P,
						d_Q,
						rand_state,
						gpu_dynamic_rate,
						cur_iter,
						num_iters,
						lambda_p,
						lambda_q,
						update_vector_size,
						max_update_count_per_block,
						update_count_per_partition[2]);
				}
				else if (part == 4) {
					CHECK(cudaMalloc((void**)&d_R, sizeof(mf_node)*tr->count4));
					CHECK(cudaMemcpy(d_R, tr->block1, sizeof(mf_node)*(tr->count4), cudaMemcpyHostToDevice));
					sgd::hogwild_train << <1, 128 >> > (
						d_R,
						tr->nnz,
						model->k,
						d_P,
						d_Q,
						rand_state,
						gpu_dynamic_rate,
						cur_iter,
						num_iters,
						lambda_p,
						lambda_q,
						update_vector_size,
						max_update_count_per_block,
						update_count_per_partition[3]);
				}
				CHECK(cudaFree(d_R));
			}
		}
		printf("Copying Back To Host\n\n");
		CHECK(cudaMemcpy(model->P, d_P, sizeof(short)*model->m*model->k, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(model->Q, d_Q, sizeof(short)*model->n*model->k, cudaMemcpyDeviceToHost));
		printf("Copy Back Successful!\n");
		printf("\n\nFreeing Device Memory");
		free(update_count_per_partition);
		CHECK(cudaFree(d_P));
		CHECK(cudaFree(d_Q));
	}
}

#endif //_SGD_GPU_H