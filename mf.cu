#include "../common/common.h"
#include <direct.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>

#include "mf.h"
#include "sgd.h"

using namespace std; 

/** @brief helper function for printing size of the file
* @param *ptr pointer to FILE
*/
void print_file_len(FILE* ptr) {
	mf_long size = 0;
	fseek(ptr, 0, SEEK_END);
	size = (mf_long)ftell(ptr);
	fseek(ptr, 0, SEEK_SET);
	printf("Size of file: %lld\n", size);
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
void partition(mf_problem& prob) {
	//blocks parts;
	printf("\n==========Partitioning=============\n");
	mf_long count1 = 0;
	mf_long count2 = 0;
	mf_long count3 = 0;
	mf_long count4 = 0;
	for (int i = 0; i < prob.nnz; i++) {
		if(i%10000000 == 0) {
			printf("Currently Partition Count Progress: %d\n", i);
		}
		int u = prob.R[i].u;
		int v = prob.R[i].v;
		if(u< prob.m/2) {
			if(v < prob.n/2) {
				count1 += 1;
			} else {
				count2 += 1;
			}
		} else {
			if(v < prob.n/2) {
				count3 += 1;
			} else {
				count4 += 1;
			}
		}
	}
	prob.count1 = count1;
	prob.count2 = count2;
	prob.count3 = count3;
	prob.count4 = count4;
	printf("\n===========Partition Counts===========\n");
	printf("Part 1: %lld\n", count1);
	printf("Part 2: %lld\n", count2);
	printf("Part 3: %lld\n", count3);
	printf("Part 4: %lld\n\n", count4);
	printf("Partition Total: %lld\n", count1 + count2 + count3 + count4);
	mf_node *b1 = new mf_node[count1];
	mf_node *b2 = new mf_node[count2];
	mf_node *b3 = new mf_node[count3];
	mf_node *b4 = new mf_node[count4];
	count1 = 0; 
	count2 = 0;
	count3 = 0; 
	count4 = 0;
	for (int i = 0; i < prob.nnz; i++) {
		long long u = prob.R[i].u;
		long long v = prob.R[i].v;

		if(u< prob.m/2) {
			if(v < prob.n/2) {
				b1[count1].u = prob.R[i].u;
				b1[count1].v = prob.R[i].v;
				b1[count1].r = prob.R[i].r;
				count1 += 1;
			} else {
				b2[count2].u = prob.R[i].u;
				b2[count2].v = prob.R[i].v;
				b2[count2].r = prob.R[i].r;
				count2 += 1;
			}
		} else {
			if(v < prob.n/2) {
				b3[count3].u = prob.R[i].u;
				b3[count3].v = prob.R[i].v;
				b3[count3].r = prob.R[i].r;
				count3 += 1;
			} else {
				b4[count4].u = prob.R[i].u;
				b4[count4].v = prob.R[i].v;
				b4[count4].r = prob.R[i].r;
				count4 += 1;
			}
		}
	}
	prob.block1 = b1;
	prob.block2 = b2;
	prob.block3 = b3;
	prob.block4 = b4;
}
/** @brief Free up memory from model
* @param mf_model ** model: model to destroy
*/
void mf_destroy_model(mf_model **model)
{
	if (model == nullptr || *model == nullptr)
		return;
#ifdef _WIN32
	_aligned_free((*model)->P);
	_aligned_free((*model)->Q);
#else
	free((*model)->P);
	free((*model)->Q);
#endif
	delete *model;
	*model = nullptr;
}

/** @brief Free up partition memory from mf_problem
*  @param mf_problem& b: structure with partitions
*/
void destroy_blocks(mf_problem & b)
{
	free(b.block1);
	free(b.block2);
	free(b.block3);
	free(b.block4);
}

/** @brief Collect mean and standard deviation from the problem to normalize

* @param mf_problem* prob: problem to analyze
* @param mf_float &avg: where to store average
* @param mf_float &std_dev: Where to store std_dev
*/
void get_mean_stddev(mf_node* R, mf_float &avg, mf_float &std_dev, mf_long nnz) {
	double tmp_mean = 0;
	double tmp_stddev = 0;

	for (long long i = 0; i < nnz; i++) {
		float rating = R[i].r;
		tmp_mean += (double)rating;
		tmp_stddev += (double)rating * rating;
	}
	tmp_mean = tmp_mean / (double)nnz;
	tmp_stddev = tmp_stddev / (double)nnz;

	avg = (mf_float)tmp_mean;
	std_dev = (mf_float)sqrt(tmp_stddev - tmp_mean*tmp_mean);
}

/* @brief Normalizes the data from stats obtained form get_mean_stddev
*
* @param mf_node* R: R matrix to normalize
* @param mf_long nnz: mf_problem's nnz. Non Zero Entries
* @param mf_float scale: How much to normalize by
*/
void normalize(mf_node* R, mf_long nnz, mf_float scale) {
	printf("Before R[0]: %f\n", R[0].r);
	for (long long i = 0; i < nnz; i++) {
		R[i].r /= scale;
	}
}

/** @brief read in matrix problem to be solved. Code designed for Netflix.bin data
*			Performs normalization as well.
*
* @param string path: path to netflix data
* @return mf_problem with R, m, n, nnz initialized
*/
mf_problem read_problem(string path)
{
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
	print_file_len(fptr);

	if (fptr == NULL) {
		printf("error file open %s\n", path.c_str());
		return prob;
	}
	fread(&prob.m, sizeof(unsigned int), 1, fptr);
	fread(&prob.n, sizeof(unsigned int), 1, fptr);
	fread(&prob.nnz, sizeof(unsigned int), 1, fptr);

	mf_node *R = new mf_node[prob.nnz];

	long long idx = 0;
	while (true)
	{
		int flag = 0;
		int u, v;
		float r;

		flag += fread(&u, sizeof(int), 1, fptr);
		flag += fread(&v, sizeof(int), 1, fptr);
		flag += fread(&r, sizeof(float), 1, fptr);

		if (flag != 3) {
			break;
		}
		R[idx].u = u;
		R[idx].v = v;
		R[idx].r = r;
		idx++;
	}
	printf("m:%lld, n:%lld, nnz:%lld\n", prob.m, prob.n, prob.nnz);

	mf_float avg;
	mf_float std_dev;
	mf_float scale = 0.0;

	get_mean_stddev(R, avg, std_dev, prob.nnz);

	printf("\n[STATISTICS] Mean: %.3f\tStd_Dev: %.3f\n", avg, std_dev);
	scale = max((mf_float)1e-4, std_dev);

	printf("Performing Normalization using: %f\n", std_dev);
	normalize(R, prob.nnz, scale);

	prob.R = R;
	fclose(fptr);

	return prob;
}

/** @brief load initial model for mf_model. Initializes P, Q matrices
* @param char const *path: path to file
* @return initialized model
*/
mf_model * mf_init_model(char const *path)
{
	printf("Loading MF Model\n");

	FILE* fptr = fopen(path, "rb");
	print_file_len(fptr);
	if (fptr == NULL)
	{
		printf("%s open failed\n", path);
		exit(0);
	}
	clock_t start = clock();

	mf_model *model = new mf_model;
	model->P = nullptr;
	model->Q = nullptr;

	int count;

	int tmp_m, tmp_n, tmp_k;

	count = fread(&tmp_m, sizeof(int), 1, fptr);
	count = fread(&tmp_n, sizeof(int), 1, fptr);
	count = fread(&tmp_k, sizeof(int), 1, fptr);

	model->m = tmp_m;
	model->n = tmp_n;
	model->k = tmp_k;

	printf("m:   %lld\n", model->m);
	printf("n:   %lld\n", model->n);
	printf("k:   %lld\n", model->k);

	printf("p_size:%lld\n", ((long long)model->m)*model->k);

	try
	{
		model->P = malloc_aligned_float<short>((long long)model->m*model->k);
		model->Q = malloc_aligned_float<short>((long long)model->n*model->k);
	}
	catch (bad_alloc const &e)
	{
		cerr << e.what() << endl;
		mf_destroy_model(&model);
		return nullptr;
	}

	auto read = [&](short *ptr, mf_int size)
	{
		for (mf_int i = 0; i < size; i++)
		{
			short *ptr1 = ptr + (long long)i*model->k;
			count = fread(ptr1, sizeof(short), model->k, fptr);
		}
	};
	
	printf("Loading P m:\t%lld\n", model->m);
	read(model->P, model->m);
	printf("loading Q n:\t%lld\n", model->n);
	read(model->Q, model->n);

	printf("time elapsed:%.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);

	return model;
}

/** @brief Print Parameters used
*/
void print_params(hog_params &params) {
	printf("\n\n=============Using Parameters============\n");
	printf("X_Part: \t%d\n", params.x_part);
	printf("Y_Part: \t%d\n", params.y_part);
	printf("batch_size: \t%d\n", params.batch_size);
	printf("max_iters: \t%d\n", params.max_iters);
	printf("lambda_p: \t%.5f\n", params.lambda_p);
	printf("lambda_q: \t%.5f\n", params.lambda_q);
	printf("Initial Alpha: \t%.5f\n", params.alpha);
}
/*
void split_grids(mf_problem& prob) {
	clock_t start;

	printf("Partition Problem\n");

	mf_long u_seg, v_seg;
	if (prob.x_part == 1) {
		u_seg = prob.m;
	} else {
		u_seg = (mf_long)ceil((double)prob.m / prob.x_part);
	}
	if (prob.y_part == 1) {
		v_seg = prob.n;
	} else {
		v_seg = (mf_long)ceil((double)prob.n / prob.y_part);
	}
	prob.u_seg_len = u_seg;
	prob.v_seg_len = v_seg;
	auto get_grid_id = [=](int u, int v) {
		return ((u / u_seg)*prob.y_part + v / v_seg);
	};
	prob.gridSize = new long long[prob.x_part*prob.y_part];
	long long *gridSize = prob.gridSize;
	for (long long i = 0; i < prob.nnz; i++) {
		int tmp_u = prob.R[i].u;
		int tmp_v = prob.R[i].v;
		gridSize[get_grid_id(tmp_u, tmp_v)]++;
	}
	long long max_grid_size = 0;
	// Fine till here
	for (int i = 0; i < prob.x_part*prob.y_part; i++) {
		if (max_grid_size < prob.gridSize[i]) {
			max_grid_size = prob.gridSize[i];
		}
	}
	prob.maxGridSize = max_grid_size;
	mf_node **R2D = new mf_node*[prob.x_part*prob.y_part + 1];
	mf_node *R = prob.R;
	R2D[0] = R;
	for (int grid = 0; grid < prob.x_part*prob.y_part; grid++) {
		R2D[grid + 1] = R2D[grid] + gridSize[grid];
	}
	prob.R2D = R2D;

	mf_node** pivots = new mf_node * [prob.x_part*prob.y_part];
	for (int i = 0; i < prob.x_part*prob.y_part; i++) {
		pivots[i] = R2D[i];
	}
	for (int grid = 0; grid < prob.x_part*prob.y_part; grid++) {
		for (mf_node* pivot = pivots[grid]; pivot != R2D[grid + 1];) {
			int corre_grid = get_grid_id(pivot->u, pivot->v);
			if (corre_grid == grid) {
				pivot++;
				continue;
			}
			mf_node *next = pivots[corre_grid];
			swap(*pivot, *next);
			pivots[corre_grid]++;
		}
	}
	printf("Done with Grid Problem\n");
	printf("\n\n");
}

*/
int main(int argc, char* argv) {
	printf("\n=============Loading Parametrs=============\n");
	hog_params params;
	print_params(params);

	printf("\n============Creating MF_Problem============\n");
	mf_problem prob_train;
	prob_train = read_problem("netflix_mm.bin");

	printf("\n============Creating Partitions============\n");
	partition(prob_train);

	//printf("\n==============Calling Grid Prob============\n");
	//split_grids(prob_train);

	printf("\n============Initializing Model=============\n");
	mf_model* model = mf_init_model("init_pqmodel_hf.bin");
	printf("\n============Starting SGD Train=============\n");
	sgd::sgd_train(&prob_train, model);

	printf("\n============Freeing Partitions=============\n");
	destroy_blocks(prob_train);

	printf("\n============ Destroying Model =============\n");
	mf_destroy_model(&model);
}

