#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <numeric>
// CPU library for Half float
#include "./half-1.12.0/half.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// modified or extracted functions from https://github.com/cuMF/cumf_sgd//mf-predict.cpp and mf.h
// in order to support half-type models
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "mf.h"
using namespace std;
using namespace mf;

void mf_destroy_model(mf_model **model);
mf_double calc_rmse(mf_problem *prob, mf_model *model);
mf_float mf_predict(mf_model const *model, mf_int u, mf_int v);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct Option
{
	Option() : eval(RMSE) {}
    string test_path, model_path, output_path;
    mf_int eval;
};

string predict_help()
{
    return string(
"usage: mf-predict [options] test_file model_file [output_file]\n"
"\n"
"options:\n"
"-e <eval>: specify the evaluation criterion (default 0)\n"
"\t 0 -- root mean square error\n"
"\t 1 -- mean absolute error\n"
"\t 2 -- generalized KL-divergence\n"
"\t 5 -- logarithmic error\n"
"\t 6 -- accuracy\n"
"\t10 -- row-wise mean percentile rank\n"
"\t11 -- column-wise mean percentile rank\n"
"\t12 -- row-wise area under the curve\n"
"\t13 -- column-wise area under the curve\n");
}

mf_model* mf_load_model(char const *path)
{
	printf("mf_load_model called\n");

	FILE* fptr = fopen(path, "rb");
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
	prob.R = R;

	fclose(fptr);
	printf("m:%lld, n:%lld, nnz:%lld\n", prob.m, prob.n, prob.nnz);
	return prob;
}


Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    mf_int i;

	for (i = 1; i < argc; i++)
	{
		if (args[i].compare("-e") == 0)
		{
			if ((i + 1) >= argc)
				throw invalid_argument("need to specify evaluation criterion after -e");
			i++;
			option.eval = atoi(argv[i]);
			if (option.eval != RMSE &&
				option.eval != MAE &&
				option.eval != GKL &&
				option.eval != LOGLOSS &&
				option.eval != ACC &&
				option.eval != ROW_AUC &&
				option.eval != COL_AUC &&
				option.eval != ROW_MPR &&
				option.eval != COL_MPR)
				throw invalid_argument("unknown evaluation criterion");
		}
		else
			break;
	}

    if(i >= argc-1)
        throw invalid_argument("testing data and model file not specified");
    option.test_path = string(args[i++]);
    option.model_path = string(args[i++]);

    if(i < argc)
    {
        option.output_path = string(args[i]);
    }
    else if(i == argc)
    {
        const char *ptr = strrchr(&*option.test_path.begin(), '/');
        if(!ptr)
            ptr = option.test_path.c_str();
        else
            ++ptr;
        option.output_path = string(ptr) + ".out";
    }
    else
    {
        throw invalid_argument("invalid argument");
    }

    return option;
}

void predict(string test_path, string model_path, string output_path, mf_int eval)
{
    mf_problem prob = read_problem(test_path);

    mf_model *model = mf_load_model(model_path.c_str());
    if(model == nullptr)
        throw runtime_error("cannot load model from " + model_path);

	auto rmse = calc_rmse(&prob, model);
	cout << fixed << setprecision(4) << "RMSE = " << rmse << endl;

    mf_destroy_model(&model);
}

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

mf_float mf_predict(mf_model const *model, mf_int u, mf_int v)
{
	using half_float::half;

	if (u < 0 || u >= model->m || v < 0 || v >= model->n)
		return model->b;

	half *ph = (half*)model->P + ((mf_long)u)*model->k;
	half *qh = (half*)model->Q + ((mf_long)v)*model->k;

	float p, q;
	mf_float z = 0.0f;
	for (int w = 0; w < model->k; w++) {
		p = (float)(*ph);
		q = (float)(*qh);
		z += p*q;
		ph++;
		qh++;
	}

	if (isnan(z))
		z = model->b;

	if (model->fun == P_L2_MFC &&
		model->fun == P_L1_MFC &&
		model->fun == P_LR_MFC)
		z = z > 0.0f ? 1.0f : -1.0f;

	return z;
}

mf_double calc_rmse(mf_problem *prob, mf_model *model)
{
	printf("calculating rmse ...\n");
	if (prob->nnz == 0)
		return 0;
	mf_double loss = 0;

	for (mf_long i = 0; i < prob->nnz; i++)
	{
		mf_node &N = prob->R[i];
		mf_float e = N.r - mf_predict(model, N.u, N.v);

		loss += e*e;

		if (i % 100000000 == 0 && i > 0)printf("progress: %%%.3lf, est_RMSE: %.4lf\n", ((double)100.0)*i / prob->nnz, sqrt(loss / (i + 1)));
	}
	return sqrt(loss / prob->nnz);
}

int main(int argc, char **argv)
{
    Option option;
    try
    {
        option = parse_option(argc, argv);
    }
    catch(invalid_argument &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    try
    {
        predict(option.test_path, option.model_path, option.output_path, option.eval);
    }
    catch(runtime_error &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    return 0;
}
