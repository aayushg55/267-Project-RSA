#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>
#include <cmath>
#include <cuda.h>
#include "butil.hpp"
#include "cublas_v2.h"
#include <curand.h>
#include <math.h>
#if !UPCXX_KIND_CUDA
#error "This example requires UPC++ to be built with CUDA support."
#endif

using namespace std;
using namespace upcxx;
using gp_device = global_ptr<double, gpu_default_device::kind>;

int main(int argc, char** argv) {
    init();
    
    int size = 1;
    int my_rank = rank_me();
    int num_procs = rank_n();

    size_t segsize = 1024*1024*1024;         // 4 MiB

    auto gpu_alloc = make_gpu_allocator<cuda_device>(segsize);
    UPCXX_ASSERT_ALWAYS(gpu_alloc.is_active(), "Failed to open GPU:\n");

    size_t seq_length = 720;
    size_t local_seq_length = ceil(seq_length/(num_procs));
    size_t qk_dim = local_seq_length;
    size_t v_dim = local_seq_length;

    size_t partition_size = local_seq_length * qk_dim;
    size_t seq_seq = local_seq_length * local_seq_length;
    size_t full_mat_size = seq_length * qk_dim;

    // Set the seed for the random number generator using the system clock
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // allocate memory on GPU (per process)
    gp_device my_gpu_q = gpu_alloc.allocate<double>(partition_size);
    gp_device my_gpu_k = gpu_alloc.allocate<double>(partition_size);
    gp_device my_gpu_v = gpu_alloc.allocate<double>(partition_size);

    gp_device gpu_attn_scores = gpu_alloc.allocate<double>(seq_seq * num_procs);
    gp_device gpu_out_scores = gpu_alloc.allocate<double>(local_seq_length*v_dim);

    // create dist object from all gpu arrays
    // dist_object<gp_device> dobj_q(gpu_array);
    dist_object<gp_device> dobj_k(my_gpu_k);
    dist_object<gp_device> dobj_v(my_gpu_v);

    //initialize gpu q,k,v

    barrier();

    // cout << "my rank: " << my_rank << " , total procs: " << num_procs << "\n" << flush;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    // size_t m = local_matrix_len;
    // size_t lda = m;
    const double alp = 1;
    const double bet = 0;
    const double *alpha = &alp;
    const double *beta = &bet;

    double* d_my_q = gpu_alloc.local(my_gpu_q);
    double* d_my_k = gpu_alloc.local(my_gpu_k);
    double* d_my_v = gpu_alloc.local(my_gpu_v);

    double* d_attn_scores = gpu_alloc.local(gpu_attn_scores);

    curandSetPseudoRandomGeneratorSeed(prng, my_rank);

    // Fill the array with random numbers on the device
    curandGenerateUniformDouble(prng, d_my_q, partition_size);
    curandGenerateUniformDouble(prng, d_my_k, partition_size);
    curandGenerateUniformDouble(prng, d_my_v, partition_size);
    cudaDeviceSynchronize();
    global_ptr<double> host_array_attn = new_array<double>(seq_seq*num_procs);
    double* host_attn = host_array_attn.local();

    upcxx::barrier();
    gp_device gpu_recv_buff = gpu_alloc.allocate<double>(partition_size);
    global_ptr<double> host_recv_buff = new_array<double>(partition_size);
    auto start_time = std::chrono::steady_clock::now();
/****************************************************************************/
    // RingQK
    gp_device neighbor_gpu_k;

    for (int count = 0; count < num_procs*100; ++count) {
        int next = (my_rank + count) % num_procs;
        neighbor_gpu_k = dobj_k.fetch(next).wait();
        upcxx::copy(neighbor_gpu_k, gpu_recv_buff, partition_size).wait();

        double* d_other_k = gpu_alloc.local(gpu_recv_buff);
        
        //compute my_q * sub_k_t
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, local_seq_length, qk_dim, alpha,
                    d_my_q, local_seq_length, 
                    d_other_k, local_seq_length, beta, 
                    d_attn_scores+seq_seq*next, local_seq_length);
        cudaDeviceSynchronize();
    }

    // copy(gpu_attn_scores, host_array_attn, local_matrix_size*num_procs).wait();
/****************************************************************************/

    //compute softmax over entire attn_scores matrix

    barrier();
/****************************************************************************/
    // // RingAV

    double* d_my_out = gpu_alloc.local(gpu_out_scores);
    gp_device neighbor_gpu_v;

    for (int count = 0; count < num_procs*100; ++count) {
        int next = (my_rank + count) % num_procs;
        neighbor_gpu_v = dobj_v.fetch(next).wait();

        upcxx::copy(neighbor_gpu_v, gpu_recv_buff, partition_size).wait();
        
        //compute sub_A * sub_v
        double* d_other_v = gpu_alloc.local(gpu_recv_buff);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, v_dim, local_seq_length, alpha,
                d_attn_scores+seq_seq*next, local_seq_length, 
                d_other_v, local_seq_length, alpha, 
                d_my_out, local_seq_length);
        cudaDeviceSynchronize();
    }
    barrier();

/****************************************************************************/
    //backwards for AV
    // grad_output of dim local * v_dim; attn map of dim local * seq
    //m,n,k are post transpose

    gp_device grad_output = gpu_alloc.allocate<double>(local_seq_length*v_dim);
    double* grad_output_raw = gpu_alloc.local(grad_output);

    size_t grad_v_size = seq_length * v_dim;
    double* d_grad_v;
    cudaMalloc((void**)&d_grad_v, grad_v_size * sizeof(double));

    curandGenerateUniformDouble(prng, grad_output_raw, local_seq_length*v_dim);

    //compute grad_v = Attn_T * grad_output
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_length, v_dim, local_seq_length, alpha,
                d_attn_scores, local_seq_length, 
                grad_output_raw, seq_length, beta, 
                d_grad_v, seq_length);

    // reduce grad_v
    double* host_grad_v = (double*) malloc(grad_v_size * sizeof(double));
    cudaMemcpy(host_grad_v, d_grad_v, grad_v_size, cudaMemcpyDeviceToHost);
    upcxx::reduce_all(host_grad_v, host_grad_v, grad_v_size, upcxx::op_fast_add, upcxx::world()).wait();
    cudaMemcpy(d_grad_v, host_grad_v, grad_v_size, cudaMemcpyHostToDevice);

    size_t grad_attn_scores_size = local_seq_length*seq_length;
    double* d_grad_attn_scores;
    cudaMalloc((void**)&d_grad_attn_scores, grad_attn_scores_size * sizeof(double));

    for (int count = 0; count < num_procs*100; ++count) {
        int next = (my_rank + count) % num_procs;
        neighbor_gpu_v = dobj_v.fetch(next).wait();

        upcxx::copy(neighbor_gpu_v, gpu_recv_buff, partition_size).wait();
        double* d_other_v = gpu_alloc.local(gpu_recv_buff);

        //compute grad_output * sub_v_T
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, local_seq_length, v_dim, alpha,
                grad_output_raw, local_seq_length, 
                d_other_v, local_seq_length, 
                alpha, 
                d_grad_attn_scores + next*seq_seq, local_seq_length);
        cudaDeviceSynchronize();
    }
    barrier();
/****************************************************************************/

    //backward pass for QK_T
    // gp_device grad_output = gpu_alloc.allocate<double>(local_seq_length*seq_length);
    // double* grad_output_raw = gpu_alloc.local(grad_output);

    // gp_device grad_q = gpu_alloc.allocate<double>(partition_size);
    // double* grad_q_raw = gpu_alloc.local(grad_q);

    // double* grad_k_raw;
    // cudaMalloc((void**)&grad_k_raw, seq_length * qk_dim * sizeof(double));
    
    // // compute local grad_k contribution
    // cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_length, qk_dim, local_seq_length, alpha,
    //             d_grad_attn_scores, local_seq_length, 
    //             d_my_q, local_seq_length, beta, 
    //             grad_k_raw, seq_length);

    // //reduce grad_k over all processors
    // double* host_grad_k = (double*) malloc(seq_length * qk_dim* sizeof(double));
    // cudaMemcpy(host_grad_k, grad_k_raw, seq_length * qk_dim, cudaMemcpyDeviceToHost);
    // upcxx::reduce_all(host_grad_k, host_grad_k, seq_length * qk_dim, upcxx::op_fast_add, upcxx::world()).wait();
    // cudaMemcpy(grad_k_raw, host_grad_k, seq_length * qk_dim, cudaMemcpyHostToDevice);

    // for (int count = 0; count < num_procs*100; ++count) {
    //     int next = (my_rank + count) % num_procs;

    //     neighbor_gpu_k = dobj_k.fetch(next).wait();
    //     upcxx::copy(neighbor_gpu_k, gpu_recv_buff, partition_size).wait();
    //     double* d_other_k = gpu_alloc.local(gpu_recv_buff);
        
    //     //compute grad_output[start:end] * sub_k
    //     cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, local_seq_length, local_seq_length, alpha,
    //                 d_grad_attn_scores + next*seq_seq, local_seq_length,
    //                 d_other_k, local_seq_length,
    //                 alpha,
    //                 grad_q_raw, local_seq_length);
    //     cudaDeviceSynchronize();
    // }

    /**********************************************************************************/
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    //     // Finalize
    if (my_rank == 0) {
        cout << "Time = " << seconds << " seconds\n" << flush;
    }

    delete_array(host_recv_buff);
    
    // Free gpu memory
    cublasDestroy(handle);

    gpu_alloc.deallocate(my_gpu_q);
    gpu_alloc.deallocate(my_gpu_k);
    gpu_alloc.deallocate(my_gpu_v);
    gpu_alloc.deallocate(gpu_attn_scores);
    gpu_alloc.deallocate(gpu_out_scores);
    gpu_alloc.deallocate(gpu_recv_buff);
    // gpu_alloc.deallocate(grad_k);
    // gpu_alloc.deallocate(grad_q);
    
    // cudaFree(grad_k_raw);

    gpu_alloc.destroy();
    cout << "SUCCESS" << std::endl;

    finalize();
    return 0;
}
