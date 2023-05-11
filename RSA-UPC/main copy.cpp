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

#if !UPCXX_KIND_CUDA
#error "This example requires UPC++ to be built with CUDA support."
#endif

using namespace std;
using namespace upcxx;

int main(int argc, char** argv) {
    init();
    
    int size = 1;
    int my_rank = rank_me();
    int num_procs = rank_n();

    size_t segsize = 300*1024*1024;         // 4 MiB

    auto gpu_alloc = make_gpu_allocator<cuda_device>(segsize);
    UPCXX_ASSERT_ALWAYS(gpu_alloc.is_active(), "Failed to open GPU:\n");

    size_t seq_length = 1024;
    size_t local_seq_length = seq_length/num_procs;
    size_t qk_dim = 1024;
    size_t v_dim = 1024;

    size_t partition_size = local_seq_length * qk_dim;
    size_t seq_seq = local_seq_length * local_seq_length;

    // allocate memory on GPU (per process)
    global_ptr<double,gpu_default_device::kind> my_gpu_q = gpu_alloc.allocate<double>(partition_size);
    global_ptr<double,gpu_default_device::kind> my_gpu_k = gpu_alloc.allocate<double>(partition_size);
    global_ptr<double,gpu_default_device::kind> my_gpu_v = gpu_alloc.allocate<double>(partition_size);

    global_ptr<double,gpu_default_device::kind> gpu_attn_scores = gpu_alloc.allocate<double>(seq_seq * num_procs);
    global_ptr<double,gpu_default_device::kind> gpu_out_scores = gpu_alloc.allocate<double>(seq_seq);

    // create dist object from all gpu arrays
    // dist_object<global_ptr<double,gpu_default_device::kind>> dobj_q(gpu_array);
    dist_object<global_ptr<double,gpu_default_device::kind>> dobj_k(my_gpu_k);
    dist_object<global_ptr<double,gpu_default_device::kind>> dobj_v(my_gpu_v);

    //initialize gpu q,k,v

    // // create local array, initialize, and copy to gpu
    // global_ptr<double> host_array1 = new_array<double>(1024);
    // double *h1 = host_array1.local();
    // for (int i=0; i< 1024; i++) h1[i] = my_rank; //initialize h1

    // copy(host_array1, gpu_array, 1024).wait();


    // initialize a second array to copy neighbor gpu data into
    // global_ptr<double> host_array2 = new_array<double>(1024);
    // copy(other_gpu_array, host_array2, 1024).wait();

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

    // Set the seed for the random number generator using the system clock
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, my_rank);

    // Fill the array with random numbers on the device
    curandGenerateUniformDouble(prng, d_my_q, partition_size);
    curandGenerateUniformDouble(prng, d_my_k, partition_size);
    curandGenerateUniformDouble(prng, d_my_v, partition_size);
    cudaDeviceSynchronize();
    global_ptr<double> host_array_attn = new_array<double>(seq_seq*num_procs);
    double* host_attn = host_array_attn.local();
    barrier();

    // for (int i = 0; i < local_matrix_size; i++) {
    //     host_k[i] = host_q[i] = host_v[i] = i;
    // }
    // copy(host_array_q, my_gpu_q, local_matrix_size).wait();
    // copy(host_array_k, my_gpu_k, local_matrix_size).wait();
    // copy(host_array_v, my_gpu_v, local_matrix_size).wait();

    // RingQK
    global_ptr<double,gpu_default_device::kind> neighbor_gpu_k;
    global_ptr<double,gpu_default_device::kind> gpu_recv_buff = gpu_alloc.allocate<double>(partition_size);
    global_ptr<double> host_recv_buff = new_array<double>(partition_size);

    auto start_time = std::chrono::steady_clock::now();

    for (int count = 0; count < num_procs*100; ++count) {
        int next = (my_rank + count) % num_procs;
        neighbor_gpu_k = dobj_k.fetch(next).wait();

        copy(neighbor_gpu_k, host_recv_buff, partition_size).wait();

        copy(host_recv_buff, gpu_recv_buff, partition_size).wait();

        double* d_other_k = gpu_alloc.local(gpu_recv_buff);
        
        //compute my_q * sub_k_t
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, qk_dim, local_seq_length, alpha,
                    d_my_q, local_seq_length, 
                    d_other_k, local_seq_length, beta, 
                    d_attn_scores+seq_seq*next, local_seq_length);
        cudaDeviceSynchronize();
    }

    // copy(gpu_attn_scores, host_array_attn, local_matrix_size*num_procs).wait();

    // for (int i = 0; i < 3; i++) {
    //     cout << "my rank " << my_rank << " at attn " << i << " " << host_attn[i] << "\n" << flush;
    // }

    //compute softmax over entire attn_scores matrix

    barrier();
    // // RingAV
    double* d_my_out = gpu_alloc.local(gpu_out_scores);
    global_ptr<double,gpu_default_device::kind> neighbor_gpu_v;

    for (int count = 0; count < num_procs*100; ++count) {
        int next = (my_rank + count) % num_procs;
        neighbor_gpu_v = dobj_v.fetch(next).wait();

        copy(neighbor_gpu_v, host_recv_buff, partition_size).wait();
        copy(host_recv_buff, gpu_recv_buff, partition_size).wait();
        
        //compute A * sub_v
        double* d_other_v = gpu_alloc.local(gpu_recv_buff);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, local_seq_length, v_dim, alpha,
                d_attn_scores+seq_seq*next, local_seq_length, 
                d_other_v, local_seq_length, alpha, 
                d_my_out, local_seq_length);
        cudaDeviceSynchronize();
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

        // Finalize
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
    
    gpu_alloc.destroy();
    cout << "my rank: " << my_rank << " finished destroy \n" << flush;

    finalize();
    return 0;
}
