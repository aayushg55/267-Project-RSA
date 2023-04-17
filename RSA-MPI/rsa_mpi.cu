#include "common.h"
#include <mpi.h>
#include <vector>
#include <unordered_set>
#include <bits/stdc++.h>

#include <chrono>
using namespace std::chrono;
#include <stdint.h>

using namespace std;

#define bin_size cutoff
// Put any static global variables here that you will use throughout the simulation.
vector<vector<unordered_set<particle_t*>>> grid;
vector<vector<particle_t*>> grid_above;
vector<vector<particle_t*>> grid_below;
vector<int> base;
// unordered_set<particle_t*> neighbor_bin;
vector<particle_t*> proc_particles;
int num_bins;
int height;
int height_act;
int num_procs_act;
int dirs[] = {-1, 0, 1};
vector<pair<int,int>> dir_tup = {{-1,-1},{-1,0},{-1,1}, {0,0}, {0,1}, {1,-1}, {1,0}, {1,1}};
vector<pair<int,int>> dir_tup_all = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,0}, {0,1}, {1,-1}, {1,0}, {1,1}};


typedef struct cond_particle_t {
    uint64_t id; // Particle ID
    double x;    // Position X
    double y;    // Position Y
    double vx;   // Velocity X
    double vy;   // Velocity Y
} cond_particle_t;

vector<cond_particle_t> to_send_above;
vector<cond_particle_t> to_send_below;

vector<cond_particle_t> recv_buffer_above;
vector<cond_particle_t> recv_buffer_below;

double init_time;
double comm;
double apply_force_time;
double move_time;
double rebin;
double ghost_send;
double send_clear_time;
double clear_grid_above_below;

const int nitems = 5;

MPI_Datatype CONDENSED_PARTICLE;
int blocklengths[5] = {1, 1, 1, 1, 1};
MPI_Datatype types[5] = {MPI_UINT64_T, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                            MPI_DOUBLE};
MPI_Aint offsets[5];


cond_particle_t convert(particle_t* part) {
    cond_particle_t new_part;
    new_part.id = part->id;
    new_part.x = part->x;
    new_part.y = part->y;
    new_part.vx = part->vx;
    new_part.vy = part->vy;
    return new_part;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    auto start_time = std::chrono::steady_clock::now();

    offsets[0] = offsetof(particle_t, id);
    offsets[1] = offsetof(particle_t, x);
    offsets[2] = offsetof(particle_t, y);
    offsets[3] = offsetof(particle_t, vx);
    offsets[4] = offsetof(particle_t, vy);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &CONDENSED_PARTICLE);
    MPI_Type_commit(&CONDENSED_PARTICLE);

    comm = 0;
    apply_force_time = 0;
    move_time = 0;
    rebin = 0;
    ghost_send = 0;
    init_time = 0;
    clear_grid_above_below = 0;
    send_clear_time = 0;
    
    num_procs_act = num_procs;
    num_bins = ceil(size / bin_size);
    height = floor((double)num_bins / (double)num_procs_act);

    while (height <= 6 && num_procs_act > 1) {
        num_procs_act -= 1;
        height = floor((double)num_bins / (double)num_procs_act);
    }

    base = vector<int> (num_procs_act);
    base[0] = 0;
    

    int leftover = num_bins % num_procs_act;
    for (int i = 1; i < num_procs_act; ++i) {
        base[i] = base[i - 1] + height + (i <= leftover);
    }

    height_act = height + (rank < leftover);

    // if (rank == 0)
    //     cout << "rank " << rank << " num procs " << num_procs_act << " NUM_BINS: " << num_bins << " HEIGHT " << height << " height_act" << height_act << "\n" << flush;

    if (rank < num_procs_act) {
    grid = vector<vector<unordered_set<particle_t*>>>(height_act, vector<unordered_set<particle_t*>> (num_bins));
    grid_above = vector<vector<particle_t*>> (num_bins);
    grid_below = vector<vector<particle_t*>> (num_bins);

    recv_buffer_above = vector<cond_particle_t> (num_parts);
    recv_buffer_below = vector<cond_particle_t> (num_parts);

    proc_particles.reserve(num_parts);

    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
        int idx = floor(parts[i].x / bin_size);
        int idy = floor(parts[i].y / bin_size);
        idy -= base[rank];

        if ((idy >= 0) && (idy < height_act)) {
            grid[idy][idx].insert(&parts[i]);
        } else if (idy == -1) {
            grid_below[idx].push_back(&parts[i]);
        } else if (idy == height_act) {
            grid_above[idx].push_back(&parts[i]);
        }
    }
    auto stop_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = stop_time - start_time;
    init_time += diff.count();
    }
}

void apply_force_to_both(particle_t& particle, particle_t& neighbor, bool n_is_ghost) {
    // Calculate Distance
    
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

    if (n_is_ghost)
        return;

    neighbor.ax -= coef * dx;
    neighbor.ay -= coef * dy;
    
}

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size, int rank) {
    // previous bin
    int idx = floor(p.x / bin_size);
    int idy = floor(p.y / bin_size);    
    idy -= base[rank];
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.ax = p.ay = 0;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
    
    // Re-bin
    int idx_new = floor(p.x / bin_size);
    int idy_new = floor(p.y / bin_size);
    idy_new -= base[rank];

    if ((idx != idx_new) || (idy != idy_new)) {
        grid[idy][idx].erase(&p);
        if (idy_new <= -1) {
            // send to rank below
            to_send_below.push_back(convert(&p));
            if (idy_new == -1)
                grid_below[idx].push_back(&p);
        } else if (idy_new >= height_act) {
            to_send_above.push_back(convert(&p));
            if (idy_new == height_act)
                grid_above[idx].push_back(&p);
        } else {
            // move inside the rank
            grid[idy_new][idx_new].insert(&p);
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Compute forces    

    auto stop_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = stop_time - start_time;
    apply_force_time += diff.count();

    start_time = std::chrono::steady_clock::now();

    // Clear grids before receiving new
    for (int i = 0; i < num_bins; i++) {
        grid_above[i].clear();
        grid_below[i].clear();
    }
    stop_time = std::chrono::steady_clock::now();
    diff = stop_time - start_time;
    clear_grid_above_below += diff.count();

    start_time = std::chrono::steady_clock::now();

    for (particle_t* cur : proc_particles) {
        move(*cur, size, rank); // also clears accel
    }

    proc_particles.clear();
    stop_time = std::chrono::steady_clock::now();
    diff = stop_time - start_time;
    move_time += diff.count();

    // Update to_sencd d vectors with ghost particles
    start_time = std::chrono::steady_clock::now();

    if (rank != 0) {
        for (int i = 0; i < num_bins; ++i) {
            for (particle_t* cur : grid[0][i]) {
                to_send_below.push_back(convert(cur));
            }
        }
    }

    if (rank != num_procs_act - 1) {
        for (int i = 0; i < num_bins; ++i) {
            for (particle_t* cur : grid[height_act-1][i]) {
                to_send_above.push_back(convert(cur));
            }
        }
    }
    stop_time = std::chrono::steady_clock::now();
    diff = stop_time - start_time;
    ghost_send += diff.count();
    
    int size_above = 0, size_below = 0;
    // Redistribute particles

    start_time = std::chrono::steady_clock::now();
    if (rank % 2 == 0) {
        if (rank != num_procs_act - 1) {
            MPI_Send(to_send_above.data(), to_send_above.size(), CONDENSED_PARTICLE, rank + 1, 1, MPI_COMM_WORLD);
        }
        
        if (rank != 0) {
            MPI_Send(to_send_below.data(), to_send_below.size(), CONDENSED_PARTICLE, rank - 1, 3, MPI_COMM_WORLD);
        }

        if (rank != num_procs_act - 1) {
            MPI_Status status;
            MPI_Recv(recv_buffer_above.data(), num_parts, CONDENSED_PARTICLE, rank + 1, 1, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, CONDENSED_PARTICLE, &size_above);
        }
        
        if (rank != 0) {
            MPI_Status status;
            MPI_Recv(recv_buffer_below.data(), num_parts, CONDENSED_PARTICLE, rank - 1, 3, MPI_COMM_WORLD, &status);       
            MPI_Get_count(&status, CONDENSED_PARTICLE, &size_below); 
        }

    } else {
        MPI_Status status;
        MPI_Recv(recv_buffer_below.data(), num_parts, CONDENSED_PARTICLE, rank - 1, 1, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, CONDENSED_PARTICLE, &size_below);
       
        if (rank != num_procs_act - 1) {
            MPI_Recv(recv_buffer_above.data(), num_parts, CONDENSED_PARTICLE, rank + 1, 3, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, CONDENSED_PARTICLE, &size_above);
        }
        
        MPI_Send(to_send_below.data(), to_send_below.size(), CONDENSED_PARTICLE, rank - 1, 1, MPI_COMM_WORLD); 

        if (rank != num_procs_act - 1) {
            MPI_Send(to_send_above.data(), to_send_above.size(), CONDENSED_PARTICLE, rank + 1, 3, MPI_COMM_WORLD);
        }
    }
    stop_time = std::chrono::steady_clock::now();
    diff = stop_time - start_time;
    comm += diff.count();

    start_time = std::chrono::steady_clock::now();

    // Clear to send vectors
    to_send_above.clear();
    to_send_below.clear();    
    stop_time = std::chrono::steady_clock::now();
    diff = stop_time - start_time;
    send_clear_time += diff.count();


    start_time = std::chrono::steady_clock::now();

    // Rebin the particles received
    for (int i = 0; i < size_above; i++) {
        cond_particle_t cur = recv_buffer_above[i];
        int idx = floor(cur.x / bin_size);
        int idy = floor(cur.y / bin_size);
        idy -= base[rank];
        int pos = cur.id -1;
        parts[pos].x = cur.x;
        parts[pos].y = cur.y;
        
        if (idy == height_act) {
            grid_above[idx].push_back(&parts[pos]);
        } else {
            parts[pos].vx = cur.vx;
            parts[pos].vy = cur.vy;
            grid[idy][idx].insert(&parts[pos]);
        }
    }
    
    for (int i = 0; i < size_below; i++) {
        cond_particle_t cur = recv_buffer_below[i];
        int idx = floor(cur.x / bin_size);
        int idy = floor(cur.y / bin_size);
        idy -= base[rank];
        int pos = cur.id - 1;
        parts[pos].x = cur.x;
        parts[pos].y = cur.y;

        if (idy == - 1) {
            grid_below[idx].push_back(&parts[pos]);
        } else {
            parts[pos].vx = cur.vx;
            parts[pos].vy = cur.vy;
            grid[idy][idx].insert(&parts[pos]);  
        }
    }

    stop_time = std::chrono::steady_clock::now();
    diff = stop_time - start_time;
    rebin += diff.count();
    
    }
}

bool compareParticles(particle_t p1, particle_t p2)
{
    return (p1.id < p2.id);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    vector<particle_t> vec;
    int num_process_part = 0;
    if (rank < num_procs_act) {
        for (int i = 0; i < height_act; ++i) {
            for (int j = 0; j < num_bins; ++j) {
                for (particle_t* cur: grid[i][j]) {
                    vec.push_back(*cur);
                    num_process_part++;
                }
            }
        }
    }

    int sizes[num_procs] = {0};
    MPI_Gather(&num_process_part, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int displs[num_procs];
    displs[0] = 0;
    if (rank == 0) {
        for (int i = 1; i < num_procs; i++) {
            displs[i] = sizes[i-1] + displs[i-1];
        }
    }

    MPI_Gatherv(vec.data(), vec.size(), PARTICLE, parts, sizes, displs, PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        sort(parts, parts + num_parts, compareParticles);
    }

}