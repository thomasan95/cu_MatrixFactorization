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

#include "mf.h"
#include "sgd.h"

using namespace std;
using namespace mf;
using namespace sgd;

#define pq_path "init_pqmodel_hf.bin"

int* gen_random_map(int size) {
    srand(time(NULL));
    vector<int> map(size, 0);
    for(int i = 0; i < size; i++) {
        map[i] = i;
    }
    random_shuffle(map.begin(), map.end());
    int*map_ptr = new int[size];
    for(int i = 0; i < size; i++) {
        map_ptr[i] = map[i];
    }
    return map_ptr
}

int* gen_inv_map(int* map, int size) {
    int* inv_map = new int[size];
    for( int i = 0; i < size; i++) {
        inv_map[map[i]] = i;
    }
    return inv_map;
}

void shuffle_problem(mf_problem* prob, int* p_map, int* q_map) {
    for (long long i = 0; i < prob->nnz; i++) {
        mf_node &N = prob->R[i];
        N.u = p_map[N.u];
        N.v = q_map[N.v];
    }
}

void sgd_update(Hodwild_Parameters para, mf_model *modal, mf_problem* prob, float scale) {
    curandState *rand_state;
    cudaMalloc(&rand_state, sizeof(curandState)*para.num_workers);
}

mf_model* sgd_train(mf_problem* tr, mf_problem*te, Hodwild_Parameters para) {
    float ave;
    float std_dev;
    float scale = 1.0;

    int* p_map = gen_random_map(tr->m);
    int* q_map = gen_random_map(tr->n);
    int* inv_p_map = gen_inv_map(p_map, tr->m);
    int* inv_p_map = gen_inv_map(q_map, tr->n);

    shuffle_problem(tr, p_map, q_map)

    mf_model* model = mf_load_model(pq_path);

}

