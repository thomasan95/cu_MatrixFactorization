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

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mf.h"

using namespace std;
using namespace mf;

namespace sgd
{
    struct Hogwild_Parameters
    {
        int x_part = 2;
        int y_part = 2;
        int batch_size = 128;
        int max_iter = 20;
        float lambda_p = 0.0461;
        float lambda_q = 0.0451;
        float get_alpha(int iter) {
            float alpha;
            float num = 0.08;
            float den = 1.0 + (0.3)*pow(iter, 1.5);
            alpha = num / den;
            return alpha;
	    }
    }
    mf_model* sgd_train(mf_problem*, mf_problem*, Parameter);
}

#endif