#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <cstring>
#include "./half-1.12.0/half.hpp"

#include "mf.h"
#include "sgd.h"

// #include <cuda_runetime_api.h> 
using namespace std;
using namespace mf;
using namespace sgd;

#define pq_path = "init_pqmodel_hf.bin"
#define mm_path = "netflix_mm.bin"
#define mme_path = "netflix_mme.bin"

int main(int argc, char** argv) {
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	/**
	Create mf_problem
	*/
	mf_problem prob;
	prob = read_problem(mm_file);
	/**
	Partition the Netflix Dataset for Hogwild later
	*/
	Hogwild_Parameters h_params;
	blocks parts;
	parts = partition(prob);
	/**
	initialize model
	*/
	mf_model* model = mf_load_model(pq_file);

}
