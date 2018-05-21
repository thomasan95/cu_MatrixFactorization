#include <direct.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>
#include "../common/common.h"
#include "../half-1.12.0/half.hpp"
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <random>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime_api.h>

#include "mf.h"
#include "sgd.h"

using namespace std; 

/** @brief helper function for printing size of the file
* @param *ptr pointer to FILE
*/
void print_file_len(FILE* ptr) {
	long long size = 0;
	fseek(ptr, 0, SEEK_END);
	size = (long long)ftell(ptr);
	fseek(ptr, 0, SEEK_SET);
	printf("Size of file: %lld\n", size);
}

void scale_coordinates_and_normalize(mf_problem& prob, float stddev) {
	for (long long i = 0; i < prob.nnz; i++) {
		long long tmp_u = prob.R[i].u;
		while (tmp_u >= prob.u_seg_len) {
			tmp_u = tmp_u - prob.u_seg_len;
		}
		prob.R[i].u = tmp_u;
		long long tmp_v = prob.R[i].v;
		while (tmp_v >= prob.v_seg_len) {
			tmp_v = tmp_v - prob.v_seg_len;
		}
		prob.R[i].v = tmp_v;
		prob.R[i].r = prob.R[i].r * (1.0 / stddev);
	}
}
/** @brief Function for loading Partitions after reading in the problem. These partitions
*			will be used for performing batch_Hogwild on the partition, to ensure indepedent 
*			SGD learning.
*			Will store partitions in:
*				mf_problem.count1
*				mf_problem.count2
*				mf_problem.count3
*				mf_problem.count4
*
* @param mf_problem & prob mf_problem to modify, will create partitions inside this structure
*/
void partition(mf_problem* prob, int x_part, int y_part) {
	long long u_seg_len;
	long long v_seg_len;
	u_seg_len = (long long)ceil((double)prob->m / x_part);
	v_seg_len = (long long)ceil((double)prob->n / y_part);
	
	prob->u_seg_len = u_seg_len;
	prob->v_seg_len = v_seg_len;
	

	auto get_partition_id = [=](int u, int v) {
		return ((u / u_seg_len)*prob->y_part + v / v_seg_len);
	};

	prob->counts = new long long[x_part*y_part]();
	long long *counts = prob->counts;

	for (long long i = 0; i < prob->nnz; i++) {
		int c_u = prob->R[i].u;
		int c_v = prob->R[i].v;
		counts[get_partition_id(c_u, c_v)] ++;
	}

	long long max_count = 0;
	for (int i = 0; i < x_part*y_part; i++) {
		printf("Partition %d Count: %lld\n", i, counts[i]);
		if (max_count < prob->counts[i]) {
			max_count = prob->counts[i];
		}
	}
	prob->max_count = max_count;

	// Construct Pointers to Specific Partitions of R
	mf_node** R_ptrs = new mf_node*[x_part * y_part + 1];
	mf_node* R = prob->R;
	R_ptrs[0] = R;
	for (int part = 0; part < x_part * y_part; part++) {
		R_ptrs[part + 1] = R_ptrs[part] + (long long)counts[part];
	}
	prob->R_ptrs = R_ptrs;

	// Perform Sorting of R to match partitions
	mf_node ** pivots = new mf_node*[x_part * y_part];
	for (int i = 0; i < x_part * y_part; i++) {
		pivots[i] = R_ptrs[i];
	}

	for (int part = 0; part < x_part * y_part; part++) {
		for (mf_node* pivot = pivots[part]; pivot != R_ptrs[part + 1];) {
			int c_u = pivot->u;
			int c_v = pivot->v;
			int part_id = get_partition_id(c_u, c_v);
			if (part_id == part) {
				pivot++;
				continue;
			}
			mf_node* next = pivots[part_id];
			swap(*pivot, *next);
			pivots[part_id]++;
		}
	}
}

/** @brief Collect mean and standard deviation from the problem to normalize

* @param mf_problem* prob: problem to analyze
* @param mf_float &avg: where to store average
* @param mf_float &std_dev: Where to store std_dev
*/
void get_mean_stddev(mf_problem& prob, float &avg, float &std_dev) {
	double tmp_mean = 0;
	double tmp_stddev = 0;

	for (long long i = 0; i < prob.nnz; i++) {
		float rating = prob.R[i].r;
		tmp_mean += (double)rating;
		tmp_stddev += (double)rating * rating;
	}
	tmp_mean = tmp_mean / (double)prob.nnz;
	tmp_stddev = tmp_stddev / (double)prob.nnz;

	avg = (float)tmp_mean;
	std_dev = (float)sqrt(tmp_stddev - tmp_mean*tmp_mean);
}


/** @brief read in matrix problem to be solved. Code designed for Netflix.bin data
*			Performs normalization as well.
*
* @param string path: path to netflix data
* @return mf_problem with R, m, n, nnz initialized
*/
mf_problem read_problem(string path) {
	//A simple function that reads the sparse matrix in COO manner.
	printf("\nReading Problem From:\t%s\n", path.c_str());
	mf_problem prob;
	prob.m = 1;
	prob.n = 1;
	prob.nnz = 0;
	prob.R = nullptr;
	if (path.empty()) {
		return prob;
	}
	FILE*fptr = fopen(path.c_str(), "rb");
	// Print length of the file
	//print_file_len(fptr);

	if (fptr == NULL) {
		printf("error file open %s\n", path.c_str());
		exit(0);
	}
	fread(&prob.m, sizeof(unsigned int), 1, fptr);
	fread(&prob.n, sizeof(unsigned int), 1, fptr);
	fread(&prob.nnz, sizeof(unsigned int), 1, fptr);

	printf("Prob.M: %lld, Prob.N: %lld, Prob.NNZ: %lld\n", prob.m, prob.n, prob.nnz);

	mf_node *R = new mf_node[prob.nnz];

	for(long long idx = 0; idx < prob.nnz; idx++) {
		int u, v;
		float r;

		fread(&u, sizeof(int), 1, fptr);
		fread(&v, sizeof(int), 1, fptr);
		fread(&r, sizeof(float), 1, fptr);

		R[idx].u = u;
		R[idx].v = v;
		R[idx].r = r;
	}
	printf("m:%lld, n:%lld, nnz:%lld\n", prob.m, prob.n, prob.nnz);

	prob.R = R;
	fclose(fptr);

	return prob;
}


/** @brief load initial model for mf_model. Initializes P, Q matrices
* @param char const *path: path to file
* @return initialized model
*/
mf_model * mf_init_model(char const *path, mf_problem& prob) {
	printf("Loading MF Model\n");

	FILE* fptr = fopen(path, "rb");
	//print_file_len(fptr);
	if (fptr == NULL)
	{
		printf("%s open failed\n", path);
		exit(0);
	}
	clock_t start = clock();

	//Initialize Model
	mf_model *model = new mf_model;
	model->P = nullptr;
	model->Q = nullptr;
	// Set length of each partition block inside model
	model->u_seg_len = prob.u_seg_len;
	model->v_seg_len = prob.v_seg_len;


	int m, n, k;

	fread(&m, sizeof(int), 1, fptr);
	fread(&n, sizeof(int), 1, fptr);
	fread(&k, sizeof(int), 1, fptr);

	model->m = m;
	model->n = n;
	model->k = k;
	printf("M:   %lld\n", model->m);
	printf("N:   %lld\n", model->n);
	printf("K:   %lld\n", model->k);

	model->P = (short*)malloc(m*k * sizeof(short));
	model->Q = (short*)malloc(n*k * sizeof(short));
	/*
	auto read = [&](short *ptr, long long size)
	{
		for (long long i = 0; i < size; i++)
		{
			short *ptr1 = ptr + (long long)i*model->k;
			fread(ptr1, sizeof(short), model->k, fptr);
		}
	};
	*/
	fread(model->P, sizeof(short), model->m*model->k, fptr);
	fread(model->Q, sizeof(short), model->n*model->k, fptr);
	//printf("Loading P m:\t%lld\n", model->m);
	//read(model->P, model->m);
	//printf("loading Q n:\t%lld\n", model->n);
	//read(model->Q, model->n);
	printf("Time Elapsed:\t%.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	return model;
}


/** @brief Print Parameters used
*/
void print_params(hog_params &params) {
	printf("\n=============Using Parameters============\n");
	printf("X_Part: \t%d\n", params.x_part);
	printf("Y_Part: \t%d\n", params.y_part);
	printf("batch_size: \t%d\n", params.batch_size);
	printf("max_iters: \t%d\n", params.max_iters);
	printf("lambda_p: \t%.5f\n", params.lambda_p);
	printf("lambda_q: \t%.5f\n", params.lambda_q);
	printf("Initial Alpha: \t%.5f\n", params.alpha);
}


/** @brief Function for saving the model
* @param mf_model const* model: model to save
* @param char const *path: save path
* @return 0 if success 1 if failed
*/
int save_model(mf_model const* model, char const *path) {
	printf("\n==========Saving Model ==========\n");
	clock_t start;
	start = clock();

	char command[1024];
	sprintf(command, "del %s", path);
	int sys_ret = system(command);

	FILE *f = fopen(path, "wb");
	if (f == NULL) {
		printf("Save Failed\n");
		return 1;
	}
	fwrite(&(model->m), sizeof(int), 1, f);
	fwrite(&(model->n), sizeof(int), 1, f);
	fwrite(&(model->k), sizeof(int), 1, f);

	auto write = [&](short *ptr, int size) {
		for (long long i = 0; i < size; i++) {
			short *ptr1 = ptr + (long long)i*model->k;
			fwrite(ptr1, sizeof(short), model->k, f);
		}
	};
	printf("Saving P\n");
	write(model->P, model->m);
	printf("Saving Q\n");
	write(model->Q, model->n);
	fclose(f);
	printf("Time Elapsed: %.8lfs\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	return 0;
}

void scale_model(mf_model* model, float scale) {
	float scale_factor = sqrt(scale);
	for (long long i = 0; i < model->m*model->k; i++) {
		model->P[i] *= (short)scale_factor;
	}
	for (long long i = 0; i < model->n*model->k; i++) {
		model->Q[i] *= (short)scale_factor;
	}
}

void mf_destroy_model(mf_model **model)
{
	if (model == nullptr || *model == nullptr)
		return;
	free((*model)->P);
	free((*model)->Q);
	delete *model;
	*model = nullptr;
}

int main(int argc, char* argv) {
	// Load Parameters
	hog_params params;
	print_params(params);

	printf("\n============Creating MF_Problem============\n");
	mf_problem prob_train;
	prob_train = read_problem("netflix_mm.bin");

	int x_part = params.x_part;
	int y_part = params.y_part;
	prob_train.x_part = params.x_part;
	prob_train.y_part = params.y_part;
	printf("\n============Creating Partitions============\n");
	partition(&prob_train, x_part, y_part);

	float avg;
	float std_dev;
	float scale;
	get_mean_stddev(prob_train, avg, std_dev);
	printf("\n[STATISTICS] Mean: %.3f\tStd_Dev: %.3f\n", avg, std_dev);
	scale = max((float)1e-4, std_dev);
	scale_coordinates_and_normalize(prob_train, scale);
	printf("\n============Initializing Model=============\n");
	mf_model* model = mf_init_model("init_pqmodel_hf.bin", prob_train);

	printf("\n============Starting SGD Train=============\n");
	sgd::sgd_train(&prob_train, model);
	printf("\n============ Done SGD Train ==========\n");

	printf("\n===========Scaling P and Q ==========\n");
	scale_model(model, scale);
	// Save Model
	int save;
	save = save_model(model, "trained_pq_model.bin");
	if (save == 0) {
		printf("Save Success\n");
	}
	else {
		printf("Save Failed\n");
	}
	
	free(prob_train.R);
	free(prob_train.counts);
	free(prob_train.R_ptrs);
	mf_destroy_model(&model);

	return 0;
}

