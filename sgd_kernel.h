__global__ void hogwild_train(const mf_node*R,
	long long nnz,
	int k,
	half* d_P,
	half* d_Q,
	curandState *state,
	float* dynamic_lr,
	int cur_iter,
	int num_iters,
	double lambda_p,
	double lambda_q,
	int update_count_this_block)
{
	float tmp_lr = __ldg(&dynamic_lr[cur_iter]);
	for (int update_itr = 0; update_itr < update_count_this_block; update_itr++) {
		int lane_id = threadIdx.x % 32;
		int local_wid = threadIdx.x / 32;
		int wid = 4 * blockIdx.x + local_wid;
		long long start_id = 0;
		if (lane_id == 0) {
			long long origin = (long long)(curand_uniform(&state[wid])*nnz);
			start_id = origin % nnz;
		}
		// Set Start ID for All Threads
		start_id = __shfl(start_id, 0);

		for (int i = 0; i < k; i++) {
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

			float err = r - tmp_product;

			d_P[base_p + lane_id + 0] = __float2half(tmp_p1 + tmp_lr*(err*tmp_q1 - lambda_p*tmp_p1));
			d_Q[base_q + lane_id + 0] = __float2half(tmp_q1 + tmp_lr*(err*tmp_p1 - lambda_q*tmp_q1));

			d_P[base_p + lane_id + 32] = __float2half(tmp_p2 + tmp_lr*(err*tmp_q2 - lambda_p*tmp_p2));
			d_Q[base_q + lane_id + 32] = __float2half(tmp_q2 + tmp_lr*(err*tmp_p2 - lambda_q*tmp_q2));

			d_P[base_p + lane_id + 64] = __float2half(tmp_p3 + tmp_lr*(err*tmp_q3 - lambda_p*tmp_p3));
			d_Q[base_q + lane_id + 64] = __float2half(tmp_q3 + tmp_lr*(err*tmp_p3 - lambda_q*tmp_q3));

			d_P[base_p + lane_id + 96] = __float2half(tmp_p4 + tmp_lr*(err*tmp_q4 - lambda_p*tmp_p4));
			d_Q[base_q + lane_id + 96] = __float2half(tmp_q4 + tmp_lr*(err*tmp_p4 - lambda_q*tmp_q4));
		}
	}
}