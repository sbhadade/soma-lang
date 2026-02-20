/*
 * swarm_benchmark.c — SOMA 256-agent swarm benchmark
 *
 * Demonstrates: FORK, BARRIER, BROADCAST, MSG_RECV, SOM_WALK,
 *               SOM_TRAIN, SOM_ELECT
 *
 * Measures wall-clock time for N agents doing the full training loop.
 * Run multiple times and compare output.
 *
 * Compile:
 *   gcc -O3 -march=native -o swarm swarm_benchmark.c soma_runtime.c \
 *       -lm -lpthread
 * Run:
 *   ./swarm [num_agents]   (default: 64)
 */

#include "soma_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SOMSIZE_R  16
#define SOMSIZE_C  16
#define ITERATIONS 100    /* training iterations per agent */

static soma_vm_t *vm;
static int        g_barrier_id;
static int        g_num_agents;
static atomic_int g_trained_count;

/* ── Explorer agent (the forked workers) ─────────────────────────────── */
/* 4 distinct input clusters — agents are assigned to one cluster */
static soma_vec_t clusters[4] = {
    {0.9f, 0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f, 0.1f},  /* cluster A */
    {0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f, 0.1f, 0.9f},  /* cluster B */
    {0.5f, 0.5f, 0.9f, 0.1f, 0.5f, 0.5f, 0.9f, 0.1f},  /* cluster C */
    {0.5f, 0.5f, 0.1f, 0.9f, 0.5f, 0.5f, 0.1f, 0.9f},  /* cluster D */
};

static void explorer(soma_agent_t *a) {
    /* MSG_RECV — wait for broadcast data */
    soma_msg_t msg;
    soma_msg_recv_block(a, &msg);

    /* This agent belongs to one of 4 clusters */
    int cluster_id = a->id % 4;
    int total_iters = ITERATIONS;

    for (int iter = 0; iter < total_iters; iter++) {
        /* Epoch-based decay: lr 0.5→0.01, sigma 6.0→0.8 */
        float t      = (float)iter / (float)total_iters;
        float lr     = 0.5f  * expf(-3.0f * t) + 0.01f;
        float sigma  = 6.0f  * expf(-3.0f * t) + 0.8f;

        soma_vec_t input;
        for (int d = 0; d < SOMA_VEC_DIM; d++) {
            float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
            input[d] = clusters[cluster_id][d] + noise;
            if (input[d] < 0.0f) input[d] = 0.0f;
            if (input[d] > 1.0f) input[d] = 1.0f;
        }

        /* SOM_BMU */
        int bmu_x, bmu_y;
        soma_bmu(vm, input, &bmu_x, &bmu_y);

        /* SOM_WALK toward activation */
        soma_walk_gradient(a);

        /* SOM_TRAIN */
        soma_train(vm, input, bmu_x, bmu_y, lr, sigma);

        /* Copy into R0 for this agent */
        memcpy(a->regs.R[0], input, sizeof(soma_vec_t));
        soma_epoch_inc(a);
    }

    atomic_fetch_add(&g_trained_count, 1);

    /* BARRIER — wait for all explorers to finish */
    soma_barrier_wait(vm, g_barrier_id);

    soma_agent_kill(a);
}

/* ── _start agent ────────────────────────────────────────────────────── */
static void swarm_start(soma_agent_t *a) {
    /* SOM_INIT RANDOM already done in main */

    /* Create barrier for all explorers + this agent */
    g_barrier_id = soma_barrier_new(vm, g_num_agents + 1);

    /* FORK N explorer agents */
    soma_fork(a, g_num_agents, explorer);

    /* BROADCAST data to all agents */
    soma_vec_t seed = {0.3f, 0.7f, 0.1f, 0.9f,
                       0.5f, 0.2f, 0.8f, 0.4f};
    soma_broadcast(vm, a, seed, 0xBEEF);

    printf("[_start] forked %d agents, broadcast sent\n", g_num_agents);

    /* BARRIER — wait for convergence */
    soma_barrier_wait(vm, g_barrier_id);

    printf("[_start] all %d agents converged\n",
           atomic_load(&g_trained_count));

    /* SOM_ELECT — find leader */
    soma_agent_t *leader = soma_elect(vm);
    if (leader) {
        printf("[_start] elected leader: agent %d at SOM(%d,%d)\n",
               leader->id, leader->som_x, leader->som_y);
        soma_vec_print("  leader R0", leader->regs.R[0]);
    }

    /* Show where each cluster ended up on the map */
    printf("\nCluster BMU locations (where did each data pattern settle?):\n");
    for (int ci = 0; ci < 4; ci++) {
        int bx, by;
        soma_bmu(vm, clusters[ci], &bx, &by);
        printf("  cluster %c -> SOM(%2d,%2d)  hits=%u\n",
               'A' + ci, bx, by,
               vm->som.nodes[by][bx].hit_count);
    }

    /* HALT */
    soma_agent_kill(a);
}

/* ── Benchmark harness ───────────────────────────────────────────────── */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char *argv[]) {
    g_num_agents = argc > 1 ? atoi(argv[1]) : 64;
    if (g_num_agents < 1)   g_num_agents = 1;
    if (g_num_agents > 254) g_num_agents = 254;  /* leave slots for _start */

    atomic_store(&g_trained_count, 0);

    printf("=== SOMA Swarm Benchmark ===\n");
    printf("agents=%d  som=%dx%d  iterations=%d\n\n",
           g_num_agents, SOMSIZE_R, SOMSIZE_C, ITERATIONS);

    double t0 = now_ms();

    vm = soma_vm_new(SOMSIZE_R, SOMSIZE_C);
    soma_som_init_grid(vm);   /* distributed init — better cluster separation */
    soma_spawn(vm, swarm_start);
    soma_vm_run(vm);

    double elapsed = now_ms() - t0;

    printf("\n=== Results ===\n");
    printf("wall time   : %.2f ms\n", elapsed);
    printf("agents done : %d\n",
           atomic_load(&g_trained_count));
    printf("total trains: %d\n",
           g_num_agents * ITERATIONS);
    printf("throughput  : %.0f trains/sec\n",
           (g_num_agents * ITERATIONS) / (elapsed / 1000.0));

    printf("\nSOM activation map after training:\n");
    soma_som_print_activations(vm);

    soma_vm_free(vm);
    return 0;
}
