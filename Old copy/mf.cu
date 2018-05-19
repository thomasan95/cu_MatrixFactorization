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

using namespace std; 
using namespace mf;

// #define pq_file "./data/init_pqmodel_hf.bin"
// #define mm_file "./data/netflix_mm.bin"
// #define mme_file "./data/netflix_mme.bin"


/** @brief helper function for printing size of the file
* @param *ptr pointer to FILE
*/
void printFileLength(FILE* ptr) {
	int size = 0;
	fseek(ptr, 0, SEEK_END);
	size = ftell(ptr);
	fseek(ptr, 0, SEEK_SET);
	printf("Size of file: %d\n", size);
}

mf_model* mf_load_model(char const *path)
{
	printf("Loading MF Model\n");

	FILE* fptr = fopen(path, "rb");
	printFileLength(fptr);
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

	int tmp_f, tmp_m, tmp_n, tmp_k;

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
		model->P = malloc_aligned_float<short>((mf_long)model->m*model->k);
		model->Q = malloc_aligned_float<short>((mf_long)model->n*model->k);
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
			short *ptr1 = ptr + (mf_long)i*model->k;
			count = fread(ptr1, sizeof(short), model->k, fptr);
			if (i % 100000000 == 0)printf("progress:%%%.3f\n", ((double)100.0)*i / size);
		}
	};


	printf("loading feature p m:%lld ...\n", model->m);
	read(model->P, model->m);
	printf("loading feature q n:%lld ...\n", model->n);
	read(model->Q, model->n);

	printf("time elapsed:%.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);

	return model;
}

blocks partition(mf_problem prob) {
	blocks parts;
	int count1 = count2 = count3 = count4 = 0;
	for (int i = 0; i < prob.nnz; i++) {
		if(i%100000 == 0) {
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
	mf_node *b1 = new mf_node[count1];
	mf_node *b2 = new mf_node[count2];
	mf_node *b3 = new mf_node[count3];
	mf_node *b4 = new mf_node[count4];
	count1 = 0; count2 = 0; count3 = 0; count4 = 0;
	for (int i = 0; i < prob.nnz; i++) {
		if(i%100000 == 0) {
			printf("Currently Partition Progress: %d\n", i);
		}
		int u = prob.R[i].u;
		int v = prob.R[i].v;

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
	parts.block1 = b1;
	parts.block2 = b2;
	parts.block3 = b3;
	parts.block4 = b4;
}

void destroy_blocks(blocks& b) {
	free(b.block1);
	free(b.block2);
	free(b.block3);
	free(b.block4);
}

mf_problem read_problem(string path)
{
	//A simple function that reads the sparse matrix in COO manner.
	printf("read_problem:%s\n", path.c_str());
	mf_problem prob;
	prob.m = 1;
	prob.n = 1;
	prob.nnz = 0;
	prob.R = nullptr;
	if (path.empty())
		return prob;

	FILE*fptr = fopen(path.c_str(), "rb");
	if (fptr == NULL) {
		printf("error file open %s\n", path.c_str());
		return prob;
	}
	unsigned int tmp;
	fread(&prob.m, sizeof(unsigned int), 1, fptr);
	fread(&prob.n, sizeof(unsigned int), 1, fptr);
	fread(&tmp, sizeof(unsigned int), 1, fptr);
	prob.nnz = tmp;

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

		if (flag != 3)break;

		R[idx].u = u;
		R[idx].v = v;
		R[idx].r = r;
		idx++;
	}
	printf("m:%lld, n:%lld, nnz:%lld\n", prob.m, prob.n, prob.nnz);
	// Calculate Variance and Normalize R
	float sum = 0.0;
	for (long long i = 0; i < prob.nnz; i++) {
		sum += R[i].r;
	}
	float mean = sum / prob.nnz;

	float sq_sum = 0.0;
	float tmp = 0.0;
	float var = 0.0;
	for (int i = 0; i < prob.nnz; i++) {
		tmp = pow(R[i].r - mean, 2);
		sq_sum += tmp;
	}
	var = sq_sum / prob.nnz;

	printf("Getting Variance\n");
	printf("Performing Normalization using: %f\n", var);
	for (int i = 0; i < prob.nnz; i++) {
		R[i].r /= var;
	}
	prob.R = R;
	fclose(fptr);

	return prob;
}

// int main(int argc, char** argv) {
// 	int dev = 0;
// 	cudaDeviceProp deviceProp;
// 	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
// 	printf("Using Device %d: %s\n", dev, deviceProp.name);
// 	CHECK(cudaSetDevice(dev));
// 	/**
// 	Create mf_problem
// 	*/
// 	mf_problem prob;
// 	prob = read_problem(mm_file);
// 	/**
// 	Partition the Netflix Dataset for Hogwild later
// 	*/
// 	blocks parts;
// 	parts = partition(prob);
// 	/**
// 	initialize model
// 	*/
// 	mf_model* model = mf_load_model(pq_file);

// }