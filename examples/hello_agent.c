/*
 * hello_agent.c — SOMA concurrent hello agent example
 *
 * Demonstrates: SPAWN, MSG_SEND, MSG_RECV, SOM_TRAIN, AGENT_KILL
 * Mirrors hello_agent.soma exactly, using the C runtime directly.
 *
 * Compile:
 *   gcc -O3 -march=native -o hello_agent hello_agent.c soma_runtime.c \
 *       -lm -lpthread
 * Run:
 *   ./hello_agent
 */

#include "soma_runtime.h"
#include <stdio.h>
#include <string.h>

/* Shared VM pointer */
static soma_vm_t *vm;

/* ── worker agent ──────────────────────────────────────────────────────── */
static void worker(soma_agent_t *a) {
    printf("[agent %d] started at SOM(%d,%d)\n",
           a->id, a->som_x, a->som_y);

    /* MSG_RECV — blocking wait for payload */
    soma_msg_t msg;
    soma_msg_recv_block(a, &msg);

    printf("[agent %d] received message from agent %d\n",
           a->id, msg.from);

    /* SOM_BMU — find where this input best fits on the map */
    int bmu_x, bmu_y;
    soma_bmu(vm, msg.data, &bmu_x, &bmu_y);
    a->som_x = bmu_x;
    a->som_y = bmu_y;

    /* SOM_TRAIN — update the map toward this input */
    soma_train(vm, msg.data, bmu_x, bmu_y,
               soma_lr(a), soma_sigma(a));

    /* copy trained result into R0 */
    memcpy(a->regs.R[0], msg.data, sizeof(soma_vec_t));

    printf("[agent %d] trained SOM at (%d,%d)\n",
           a->id, bmu_x, bmu_y);
    soma_vec_print("  R0", a->regs.R[0]);

    /* MSG_SEND back to parent — signal done */
    soma_vec_t done = {0};
    soma_msg_send(&vm->agents[0], a, done, 0x00);

    soma_agent_kill(a);
}

/* ── main / _start agent ───────────────────────────────────────────────── */
static void start_agent(soma_agent_t *a) {
    printf("[agent %d] _start\n", a->id);

    /* SPAWN worker */
    soma_agent_t *w = soma_spawn(vm, worker);
    if (!w) { fprintf(stderr, "SPAWN failed\n"); return; }

    /* SOM_MAP — place worker at (0,0) */
    w->som_x = 0;
    w->som_y = 0;

    /* Prepare payload vector */
    soma_vec_t payload = {0.8f, 0.2f, 0.6f, 0.4f,
                          0.9f, 0.1f, 0.7f, 0.3f};

    /* MSG_SEND payload to worker */
    soma_msg_send(w, a, payload, 0xFF42);
    printf("[agent %d] sent payload to worker %d\n", a->id, w->id);

    /* WAIT — block until worker signals done */
    soma_msg_t reply;
    soma_msg_recv_block(a, &reply);
    printf("[agent %d] worker done. HALT.\n", a->id);

    soma_agent_kill(a);
}

int main(void) {
    printf("=== SOMA Hello Agent (concurrent runtime) ===\n\n");

    vm = soma_vm_new(4, 4);
    soma_som_init_random(vm);

    /* Launch _start agent */
    soma_spawn(vm, start_agent);

    /* Block until all agents finish */
    soma_vm_run(vm);

    printf("\nFinal SOM activations:\n");
    soma_som_print_activations(vm);

    soma_vm_free(vm);
    return 0;
}
