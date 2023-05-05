#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>
#include <cuda.h>
#include "cublas_v2.h"
#include <curand.h>

using namespace std;
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

MPI_Datatype PARTICLE;

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of particles" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", "");
    std::ofstream fsave(savename);

    // Init MPI
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "num_procs: " << num_procs << "\n" << std::flush;
    // Create MPI Particle Type

    // cudaStreamCreate(&stream_on_gpu_0);
    // cudaSetDevice(0);
    // cudaDeviceEnablePeerAccess( 1, 0 );
    // cudaMalloc(d_0, num_bytes); /* create array on GPU 0 */
    // cudaSetDevice(1);
    // cudaMalloc(d_1, num_bytes); /* create array on GPU 1 */
    // cudaMemcpyPeerAsync(d_0, 0, d_1, 1, num_bytes, stream_on_gpu_0);
    /* copy d_1 from GPU 1 to d_0 on GPU 0: pull copy */ 
    
    size_t seq_length = 1024;
    size_t local_seq_length = ceil(seq_length/sqrt(num_procs));
    // size_t local_seq_length = ceil(seq_length);

    size_t qk_dim = local_seq_length;
    size_t v_dim = local_seq_length;

    size_t partition_size = local_seq_length * qk_dim;
    size_t seq_seq = local_seq_length * local_seq_length;

    double* d_my_q;
    double* d_my_k;
    double* d_my_v;
    double* d_attn_scores;
    double* d_my_out;

    cudaMalloc((void**)&d_my_q, partition_size * sizeof(double)); /* create array on GPU 0 */
    cudaMalloc((void**)&d_my_k, partition_size * sizeof(double)); /* create array on GPU 0 */
    cudaMalloc((void**)&d_my_v, partition_size * sizeof(double)); /* create array on GPU 0 */

    cudaMalloc((void**)&d_attn_scores, seq_seq * num_procs * sizeof(double)); /* create array on GPU 0 */
    cudaMalloc((void**)&d_my_out, seq_seq * sizeof(double)); /* create array on GPU 0 */

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, rank);
    
    curandGenerateUniformDouble(prng, d_my_q, partition_size);
    curandGenerateUniformDouble(prng, d_my_k, partition_size);
    curandGenerateUniformDouble(prng, d_my_v, partition_size);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alp = 1;
    const double bet = 0;
    const double *alpha = &alp;
    const double *beta = &bet;


    double* rec_buff_d;
    double* send_buff_d;

    cudaMalloc((void**)&rec_buff_d, partition_size * sizeof(double)); /* create array on GPU 0 */
    cudaMalloc((void**)&send_buff_d, partition_size * sizeof(double)); /* create array on GPU 0 */
    MPI_Barrier(MPI_COMM_WORLD);

    auto start_time = std::chrono::steady_clock::now();
/*********************************************************************************************/
    // Ring QK
    // copy my_k into send_buff
    for (int iter = 0; iter < 100; iter++) {
        cudaMemcpy(send_buff_d, d_my_k, partition_size, cudaMemcpyDeviceToDevice);
        // calculate own q * k_T
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, qk_dim, local_seq_length, alpha,
                        d_my_q, local_seq_length, 
                        d_my_k, local_seq_length, beta, 
                        d_attn_scores+seq_seq*rank, local_seq_length);

        for (int i = 0; i < num_procs-1; i++) {
            int n_r = (rank+1) % num_procs;
            int n_l = (rank-1+num_procs) % num_procs;

            double* d_other_k;
            if(i % 2 == 0) {
                // send from send_buff and recv new sub_k into recv
                if (rank % 2 == 0) {
                    MPI_Send(send_buff_d, partition_size, MPI_DOUBLE, n_r, 1, MPI_COMM_WORLD);
                    MPI_Recv(rec_buff_d, partition_size, MPI_DOUBLE, n_l % num_procs, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(rec_buff_d, partition_size, MPI_DOUBLE, n_l, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(send_buff_d, partition_size, MPI_DOUBLE, n_r, 2, MPI_COMM_WORLD); 
                }
                d_other_k = rec_buff_d;
            } else {
                // send from recv_buff and recv new sub_k into send
                if (rank % 2 == 0) {
                    MPI_Send(rec_buff_d, partition_size, MPI_DOUBLE, n_r, 1, MPI_COMM_WORLD);
                    MPI_Recv(send_buff_d, partition_size, MPI_DOUBLE, n_l % num_procs, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(send_buff_d, partition_size, MPI_DOUBLE, n_l, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(rec_buff_d, partition_size, MPI_DOUBLE, n_r, 2, MPI_COMM_WORLD); 
                }
                d_other_k = send_buff_d;
            }

            //compute my_q * sub_k_t
            size_t neighbor_rank = (rank-i+num_procs) % num_procs;
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, local_seq_length, qk_dim, local_seq_length, alpha,
                        d_my_q, local_seq_length, 
                        d_other_k, local_seq_length, beta, 
                        d_attn_scores+seq_seq*neighbor_rank, local_seq_length);
            cudaDeviceSynchronize();    
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
/*********************************************************************************************/
    // compute softmax on A



/*********************************************************************************************/
    // Ring AV
    // copy my_v into send_buff
    for (int iter = 0; iter < 100; iter++) {
        cudaMemcpy(send_buff_d, d_my_v, partition_size, cudaMemcpyDeviceToDevice);
        // calculate own A * V
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, local_seq_length, v_dim, alpha,
                d_attn_scores+seq_seq*rank, local_seq_length, 
                d_my_v, local_seq_length, beta, 
                d_my_out, local_seq_length);
        cudaDeviceSynchronize();

        for (int i = 0; i < num_procs-1; i++) {
            int n_r = (rank+1) % num_procs;
            int n_l = (rank-1+num_procs) % num_procs;

            double* d_other_v;
            if(i % 2 == 0) {
                // send from send_buff and recv new sub_v into recv
                if (rank % 2 == 0) {
                    MPI_Send(send_buff_d, partition_size, MPI_DOUBLE, n_r, 1, MPI_COMM_WORLD);
                    MPI_Recv(rec_buff_d, partition_size, MPI_DOUBLE, n_l % num_procs, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(rec_buff_d, partition_size, MPI_DOUBLE, n_l, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(send_buff_d, partition_size, MPI_DOUBLE, n_r, 2, MPI_COMM_WORLD); 
                }
                d_other_v = rec_buff_d;
            } else {
                // send from recv_buff and recv new sub_v into send
                if (rank % 2 == 0) {
                    MPI_Send(rec_buff_d, partition_size, MPI_DOUBLE, n_r, 1, MPI_COMM_WORLD);
                    MPI_Recv(send_buff_d, partition_size, MPI_DOUBLE, n_l % num_procs, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(send_buff_d, partition_size, MPI_DOUBLE, n_l, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(rec_buff_d, partition_size, MPI_DOUBLE, n_r, 2, MPI_COMM_WORLD); 
                }
                d_other_v = send_buff_d;
            }

            //compute my_a * sub_v
            size_t neighbor_rank = (rank-i+num_procs) % num_procs;
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_seq_length, local_seq_length, v_dim, alpha,
                    d_attn_scores+seq_seq*neighbor_rank, local_seq_length, 
                    d_other_v, local_seq_length, alpha, 
                    d_my_out, local_seq_length);
            cudaDeviceSynchronize();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
/*********************************************************************************************/
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    double* host = (double*) malloc(seq_seq * sizeof(double));
    cudaMemcpy(host, d_my_out, seq_seq * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "my rank: " << rank << "received at 0 " << host[0] << "\n" << std::flush;
    std::cout << "my rank: " << rank << "received at 1 " << host[1] << "\n" << std::flush;

    if (rank == 0) {
        cout << "Time = " << seconds << " seconds\n" << flush;
    }
    cudaFree(rec_buff_d);
    cudaFree(send_buff_d);
    cudaFree(d_my_q);
    cudaFree(d_my_k);
    cudaFree(d_my_v);
    cudaFree(d_attn_scores);
    cudaFree(d_my_out);


    free(host);


    // Initialize Particles
   
    // MPI_Bcast(parts, num_parts, PARTICLE, 0, MPI_COMM_WORLD);

    // Algorithm
    // auto start_time = std::chrono::steady_clock::now();

    // auto end_time = std::chrono::steady_clock::now();

    // std::chrono::duration<double> diff = end_time - start_time;
    // double seconds = diff.count();

    // Finalize
    std::cout << rank << " Finished RSA!\n" << std::flush;

    // if (rank == 1) {
    //     std::cout << "Init Time = " << init_time << " seconds for " << num_parts
    //               << " particles.\n" << std::flush;
    //     std::cout << "Apply force Time = " << apply_force_time << " seconds for " << num_parts
    //               << " particles.\n" << std::flush;
    //     std::cout << "Clear above/below grids = " << clear_grid_above_below << " seconds for " << num_parts
    //               << " particles.\n" << std::flush;
      
    //     std::cout << "Move Time = " << move_time << " seconds for " << num_parts
    //             << " particles.\n" << std::flush;
    //     std::cout << "Ghost send setup time = " << ghost_send << " seconds for " << num_parts
    //             << " particles.\n" << std::flush;
    //     std::cout << "Communication Time = " << comm << " seconds for " << num_parts
    //             << " particles.\n" << std::flush;    
    //     std::cout << "Clear send vectors Time = " << send_clear_time << " seconds for " << num_parts
    //               << " particles.\n" << std::flush;
    //     std::cout << "Rebin Time = " << rebin << " seconds for " << num_parts
    //               << " particles.\n\n" << std::flush;
    // }
    MPI_Finalize();
}
