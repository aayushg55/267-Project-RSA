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

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}


int main(int argc, char** argv) {
    init();

    int my_rank = rank_me();
    int num_procs = rank_n();
    size_t segsize = 1024*1024*1024;         // 4 MiB

    auto gpu_alloc = make_gpu_allocator<cuda_device>(segsize);
    UPCXX_ASSERT_ALWAYS(gpu_alloc.is_active(), "Failed to open GPU:\n");
    
    int NUM_ITER = find_int_arg(argc, argv, "-i", 1);
    bool do_backwards = find_int_arg(argc, argv, "-b", true);
    size_t seq_length = find_int_arg(argc, argv, "-n", 1440);
    size_t local_seq_length = ceil(seq_length/(num_procs));
    size_t qk_dim = 512;
    size_t v_dim = 512;

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
    dist_object<gp_device> dobj_k(my_gpu_k);
    dist_object<gp_device> dobj_v(my_gpu_v);

    barrier();

    
    cublasHandle_t handle;
    cublasCreate(&handle);

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

    for (int count = 0; count < num_procs*NUM_ITER; ++count) {
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
    // RingAV
    double* d_my_out = gpu_alloc.local(gpu_out_scores);
    gp_device neighbor_gpu_v;

    for (int count = 0; count < num_procs*NUM_ITER; ++count) {
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
    // grad_upstream of dim local * v_dim; attn map of dim local * seq
    //m,n,k are post transpose
if (do_backwards) {
    gp_device grad_upstream = gpu_alloc.allocate<double>(local_seq_length*v_dim);
    double* grad_upstream_raw = gpu_alloc.local(grad_upstream);

    size_t grad_v_size = seq_length * v_dim;

    gp_device gp_grad_v = gpu_alloc.allocate<double>(grad_v_size);
    dist_object<gp_device> dobj_grad_v(gp_grad_v);
    double* d_grad_v = gpu_alloc.local(gp_grad_v);

    gp_device grad_v_recv_buff = gpu_alloc.allocate<double>(grad_v_size);

    size_t grad_attn_scores_size = local_seq_length*seq_length;
    double* d_grad_attn_scores;
    cudaMalloc((void**)&d_grad_attn_scores, grad_attn_scores_size * sizeof(double));

    gp_device gp_grad_q = gpu_alloc.allocate<double>(partition_size);
    double* grad_q_raw = gpu_alloc.local(gp_grad_q);

    size_t grad_k_size = seq_length * qk_dim;
    gp_device gp_grad_k = gpu_alloc.allocate<double>(grad_k_size);
    dist_object<gp_device> dobj_grad_k(gp_grad_k);
    double* d_grad_k = gpu_alloc.local(gp_grad_k);

    gp_device grad_k_recv_buff = gpu_alloc.allocate<double>(grad_k_size);

for (int j = 0; j < NUM_ITER; j++) {
    curandGenerateUniformDouble(prng, grad_upstream_raw, local_seq_length*v_dim);

    //compute grad_v = Attn_T * grad_upstream
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_length, v_dim, local_seq_length, alpha,
                d_attn_scores, local_seq_length, 
                grad_upstream_raw, local_seq_length, beta, 
                d_grad_v, seq_length);

    // reduce grad_v
    // d_grad_v += d_other_grad_v (reduce in ring fashion)
    for (int count = 1; count < num_procs; ++count) {
        int next = (my_rank + count) % num_procs;

        gp_device neighbor_grad_v = dobj_grad_v.fetch(next).wait();
        upcxx::copy(neighbor_grad_v, grad_v_recv_buff, partition_size).wait();
        double* d_other_grad_v = gpu_alloc.local(grad_v_recv_buff);
        cublasDaxpy(handle, grad_v_size, alpha, d_other_grad_v, 1, d_grad_v, 1);
        cudaDeviceSynchronize();
    }


    for (int count = 0; count < num_procs; ++count) {
        int next = (my_rank + count) % num_procs;
        neighbor_gpu_v = dobj_v.fetch(next).wait();

        upcxx::copy(neighbor_gpu_v, gpu_recv_buff, partition_size).wait();
        double* d_other_v = gpu_alloc.local(gpu_recv_buff);

        //compute grad_upstream * sub_v_T
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, local_seq_length, v_dim, alpha,
                grad_upstream_raw, local_seq_length, 
                d_other_v, local_seq_length, alpha, 
                d_grad_attn_scores + next*seq_seq, local_seq_length);
        cudaDeviceSynchronize();
    }

    barrier();
/****************************************************************************/

    //backward pass for QK_T

    // compute local grad_k contribution
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_length, qk_dim, local_seq_length, alpha,
                d_grad_attn_scores, local_seq_length, 
                d_my_q, local_seq_length, beta, 
                d_grad_k, seq_length);

    //reduce grad_k over all processors
    // d_grad_k += d_other_grad_k (reduce in ring fashion)
    for (int count = 1; count < num_procs; ++count) {
        int next = (my_rank + count) % num_procs;

        gp_device neighbor_grad_k = dobj_grad_k.fetch(next).wait();
        upcxx::copy(neighbor_grad_k, grad_k_recv_buff, partition_size).wait();
        double* d_other_grad_k = gpu_alloc.local(grad_k_recv_buff);
        cublasDaxpy(handle, grad_k_size, alpha, d_other_grad_k, 1, d_grad_k, 1);
        cudaDeviceSynchronize();
    }

    for (int count = 0; count < num_procs; ++count) {
        int next = (my_rank + count) % num_procs;

        neighbor_gpu_k = dobj_k.fetch(next).wait();
        upcxx::copy(neighbor_gpu_k, gpu_recv_buff, partition_size).wait();
        double* d_other_k = gpu_alloc.local(gpu_recv_buff);
        
        //compute grad_attn_scores[start:end] * sub_k
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, qk_dim, local_seq_length, alpha,
                    d_grad_attn_scores + next*seq_seq, local_seq_length,
                    d_other_k, local_seq_length,
                    alpha,
                    grad_q_raw, local_seq_length);
        cudaDeviceSynchronize();
    }

}

    gpu_alloc.deallocate(gp_grad_v);
    gpu_alloc.deallocate(grad_v_recv_buff);
    gpu_alloc.deallocate(grad_upstream);

    gpu_alloc.deallocate(grad_k_recv_buff);
    gpu_alloc.deallocate(gp_grad_k);
    gpu_alloc.deallocate(gp_grad_q);

    cudaFree(d_grad_attn_scores);
}
    /**********************************************************************************/
    barrier();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    if (my_rank == 0) {
        cout << "Time = " << seconds << " seconds with seq_length " << seq_length << "\n" << flush;
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
    
    gpu_alloc.destroy();

    finalize();
    return 0;
}
