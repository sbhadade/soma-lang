/*
 * soma_runtime.c — SOMA Concurrent Agent Runtime Implementation
 */

#define _GNU_SOURCE
#include "soma_runtime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

/* ── Internal helpers ────────────────────────────────────────────────────── */

static float randf(void) {
    return (float)rand() / (float)RAND_MAX;
}

static float gaussian(float dist_sq, float sigma) {
    return expf(-dist_sq / (2.0f * sigma * sigma));
}

/* ── Message Queue ───────────────────────────────────────────────────────── */

static void msgq_init(soma_msgq_t *q) {
    atomic_store(&q->head, 0);
    atomic_store(&q->tail, 0);
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void msgq_destroy(soma_msgq_t *q) {
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
    pthread_mutex_destroy(&q->mu);
}

static bool msgq_push(soma_msgq_t *q, const soma_msg_t *msg) {
    pthread_mutex_lock(&q->mu);

    /* wait if full */
    while (((atomic_load(&q->head) + 1) % SOMA_MSG_QUEUE_CAP)
            == atomic_load(&q->tail)) {
        pthread_cond_wait(&q->not_full, &q->mu);
    }

    int h = atomic_load(&q->head);
    q->buf[h] = *msg;
    atomic_store(&q->head, (h + 1) % SOMA_MSG_QUEUE_CAP);

    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
    return true;
}

/* Non-blocking pop — returns false if empty */
static bool msgq_pop_try(soma_msgq_t *q, soma_msg_t *out) {
    pthread_mutex_lock(&q->mu);
    if (atomic_load(&q->head) == atomic_load(&q->tail)) {
        pthread_mutex_unlock(&q->mu);
        return false;
    }
    int t = atomic_load(&q->tail);
    *out = q->buf[t];
    atomic_store(&q->tail, (t + 1) % SOMA_MSG_QUEUE_CAP);
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mu);
    return true;
}

/* Blocking pop — waits until a message arrives */
static void msgq_pop_block(soma_msgq_t *q, soma_msg_t *out) {
    pthread_mutex_lock(&q->mu);
    while (atomic_load(&q->head) == atomic_load(&q->tail)) {
        pthread_cond_wait(&q->not_empty, &q->mu);
    }
    int t = atomic_load(&q->tail);
    *out = q->buf[t];
    atomic_store(&q->tail, (t + 1) % SOMA_MSG_QUEUE_CAP);
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

/* ── VM lifecycle ────────────────────────────────────────────────────────── */

soma_vm_t *soma_vm_new(int som_rows, int som_cols) {
    soma_vm_t *vm = calloc(1, sizeof(soma_vm_t));
    if (!vm) return NULL;

    vm->som.rows = som_rows > SOMA_MAX_SOM_DIM ? SOMA_MAX_SOM_DIM : som_rows;
    vm->som.cols = som_cols > SOMA_MAX_SOM_DIM ? SOMA_MAX_SOM_DIM : som_cols;
    pthread_rwlock_init(&vm->som.rwlock, NULL);
    pthread_mutex_init(&vm->spawn_mu, NULL);
    atomic_store(&vm->barrier_count, 0);

    /* initialise all message queues upfront */
    for (int i = 0; i < SOMA_MAX_AGENTS; i++) {
        msgq_init(&vm->agents[i].mq);
        vm->agents[i].id = -1;   /* mark as unused */
        vm->agents[i].vm = vm;
        atomic_store(&vm->agents[i].state, SOMA_AGENT_DEAD);
        /* default SOM state */
        vm->agents[i].regs.S[0] = 0.5;   /* learning rate  */
        vm->agents[i].regs.S[1] = 2.0;   /* sigma          */
        vm->agents[i].regs.S[2] = 0.0;   /* epoch          */
    }
    return vm;
}

void soma_vm_free(soma_vm_t *vm) {
    for (int i = 0; i < SOMA_MAX_AGENTS; i++) {
        msgq_destroy(&vm->agents[i].mq);
    }
    for (int i = 0; i < atomic_load(&vm->barrier_count); i++) {
        if (vm->barriers[i].initialized)
            pthread_barrier_destroy(&vm->barriers[i].bar);
    }
    pthread_rwlock_destroy(&vm->som.rwlock);
    pthread_mutex_destroy(&vm->spawn_mu);
    free(vm);
}

/* Thread entry wrapper */
static void *agent_thread_fn(void *arg) {
    soma_agent_t *a = (soma_agent_t *)arg;
    atomic_store(&a->state, SOMA_AGENT_RUNNING);
    a->entry(a);
    atomic_store(&a->state, SOMA_AGENT_DEAD);
    return NULL;
}

void soma_vm_run(soma_vm_t *vm) {
    /* Join all live agent threads */
    for (int i = 0; i < SOMA_MAX_AGENTS; i++) {
        if (vm->agents[i].id >= 0 &&
            atomic_load(&vm->agents[i].state) != SOMA_AGENT_DEAD) {
            pthread_join(vm->agents[i].thread, NULL);
        }
    }
}

/* ── Agent management ────────────────────────────────────────────────────── */

soma_agent_t *soma_spawn(soma_vm_t *vm,
                          void (*entry)(soma_agent_t *)) {
    pthread_mutex_lock(&vm->spawn_mu);

    /* Find a free slot */
    soma_agent_t *a = NULL;
    for (int i = 0; i < SOMA_MAX_AGENTS; i++) {
        if (vm->agents[i].id < 0) {
            a = &vm->agents[i];
            a->id = i;
            vm->agent_count++;
            break;
        }
    }
    pthread_mutex_unlock(&vm->spawn_mu);

    if (!a) {
        fprintf(stderr, "soma: SPAWN failed — max agents reached\n");
        return NULL;
    }

    a->entry = entry;
    /* start at (0,0) — SOMA_MAP can relocate */
    a->som_x = 0;
    a->som_y = 0;
    atomic_store(&a->state, SOMA_AGENT_READY);

    pthread_create(&a->thread, NULL, agent_thread_fn, a);
    return a;
}

void soma_agent_kill(soma_agent_t *a) {
    atomic_store(&a->state, SOMA_AGENT_DEAD);
    a->id = -1;
    a->vm->agent_count--;
}

/* FORK: spawn N copies of the same entry function */
void soma_fork(soma_agent_t *parent, int n,
               void (*entry)(soma_agent_t *)) {
    for (int i = 0; i < n; i++) {
        soma_agent_t *child = soma_spawn(parent->vm, entry);
        if (child) {
            /* inherit parent registers */
            memcpy(&child->regs, &parent->regs, sizeof(soma_regs_t));
            /* place child near parent on SOM */
            child->som_x = (parent->som_x + i) % parent->vm->som.cols;
            child->som_y = parent->som_y;
        }
    }
}

/* ── Messaging ───────────────────────────────────────────────────────────── */

void soma_msg_send(soma_agent_t *dst, soma_agent_t *src,
                   soma_vec_t data, uint32_t tag) {
    soma_msg_t msg;
    msg.from = src ? src->id : -1;
    msg.tag  = tag;
    memcpy(msg.data, data, sizeof(soma_vec_t));
    msgq_push(&dst->mq, &msg);
}

bool soma_msg_recv(soma_agent_t *a, soma_msg_t *out) {
    return msgq_pop_try(&a->mq, out);
}

void soma_msg_recv_block(soma_agent_t *a, soma_msg_t *out) {
    atomic_store(&a->state, SOMA_AGENT_WAITING);
    msgq_pop_block(&a->mq, out);
    atomic_store(&a->state, SOMA_AGENT_RUNNING);
}

void soma_broadcast(soma_vm_t *vm, soma_agent_t *src,
                    soma_vec_t data, uint32_t tag) {
    for (int i = 0; i < SOMA_MAX_AGENTS; i++) {
        if (vm->agents[i].id >= 0 &&
            atomic_load(&vm->agents[i].state) != SOMA_AGENT_DEAD &&
            &vm->agents[i] != src) {
            soma_msg_send(&vm->agents[i], src, data, tag);
        }
    }
}

/* ── Barrier ─────────────────────────────────────────────────────────────── */

int soma_barrier_new(soma_vm_t *vm, int count) {
    int id = atomic_fetch_add(&vm->barrier_count, 1);
    pthread_barrier_init(&vm->barriers[id].bar, NULL, count);
    vm->barriers[id].count       = count;
    vm->barriers[id].initialized = true;
    return id;
}

void soma_barrier_wait(soma_vm_t *vm, int barrier_id) {
    pthread_barrier_wait(&vm->barriers[barrier_id].bar);
}

/* ── SOM Operations ──────────────────────────────────────────────────────── */

/* SOM_INIT RANDOM */
void soma_som_init_random(soma_vm_t *vm) {
    srand((unsigned)time(NULL));
    pthread_rwlock_wrlock(&vm->som.rwlock);
    for (int r = 0; r < vm->som.rows; r++) {
        for (int c = 0; c < vm->som.cols; c++) {
            for (int d = 0; d < SOMA_VEC_DIM; d++) {
                vm->som.nodes[r][c].weights[d] = randf();
            }
            vm->som.nodes[r][c].activation  = 0.0f;
            vm->som.nodes[r][c].hit_count   = 0;
        }
    }
    pthread_rwlock_unlock(&vm->som.rwlock);
}

/*
 * soma_som_init_grid — distributed initialization
 * Spreads initial weights across the SOM uniformly using
 * the first two dimensions as axes. Prevents dead neurons
 * and produces better cluster separation.
 */
void soma_som_init_grid(soma_vm_t *vm) {
    pthread_rwlock_wrlock(&vm->som.rwlock);
    for (int r = 0; r < vm->som.rows; r++) {
        for (int c = 0; c < vm->som.cols; c++) {
            float xf = (float)c / (float)(vm->som.cols - 1);
            float yf = (float)r / (float)(vm->som.rows - 1);
            for (int d = 0; d < SOMA_VEC_DIM; d++) {
                /* alternate between x and y gradients per dimension */
                vm->som.nodes[r][c].weights[d] =
                    (d % 2 == 0) ? xf : yf;
            }
            vm->som.nodes[r][c].activation = 0.0f;
            vm->som.nodes[r][c].hit_count  = 0;
        }
    }
    pthread_rwlock_unlock(&vm->som.rwlock);
}

float soma_vec_dist(soma_vec_t a, soma_vec_t b) {
    float sum = 0.0f;
    for (int i = 0; i < SOMA_VEC_DIM; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

void soma_vec_normalize(soma_vec_t v) {
    float mag = 0.0f;
    for (int i = 0; i < SOMA_VEC_DIM; i++) mag += v[i] * v[i];
    mag = sqrtf(mag);
    if (mag > 1e-9f) {
        for (int i = 0; i < SOMA_VEC_DIM; i++) v[i] /= mag;
    }
}

/* SOM_BMU — find closest node to input vector */
int soma_bmu(soma_vm_t *vm, soma_vec_t input,
             int *bmu_x, int *bmu_y) {
    float best_dist = FLT_MAX;
    int   best_r = 0, best_c = 0;

    pthread_rwlock_rdlock(&vm->som.rwlock);
    for (int r = 0; r < vm->som.rows; r++) {
        for (int c = 0; c < vm->som.cols; c++) {
            float d = soma_vec_dist(input,
                                    vm->som.nodes[r][c].weights);
            if (d < best_dist) {
                best_dist = d;
                best_r = r;
                best_c = c;
            }
        }
    }
    pthread_rwlock_unlock(&vm->som.rwlock);

    *bmu_x = best_c;
    *bmu_y = best_r;
    return best_r * vm->som.cols + best_c;
}

/* SOM_TRAIN — Kohonen update with Gaussian neighborhood */
void soma_train(soma_vm_t *vm, soma_vec_t input,
                int bmu_x, int bmu_y,
                float lr, float sigma) {
    pthread_rwlock_wrlock(&vm->som.rwlock);

    for (int r = 0; r < vm->som.rows; r++) {
        for (int c = 0; c < vm->som.cols; c++) {
            float dr = (float)(r - bmu_y);
            float dc = (float)(c - bmu_x);
            float dist_sq = dr*dr + dc*dc;
            float influence = gaussian(dist_sq, sigma);

            if (influence < 1e-4f) continue;

            soma_node_t *node = &vm->som.nodes[r][c];
            for (int d = 0; d < SOMA_VEC_DIM; d++) {
                node->weights[d] += lr * influence
                                    * (input[d] - node->weights[d]);
            }
            /* track activation as influence at this node */
            if (influence > node->activation)
                node->activation = influence;
        }
    }
    vm->som.nodes[bmu_y][bmu_x].hit_count++;
    pthread_rwlock_unlock(&vm->som.rwlock);
}

/* SOM_WALK — move agent toward highest activation neighbor */
void soma_walk_gradient(soma_agent_t *a) {
    soma_vm_t *vm = a->vm;
    int cx = a->som_x, cy = a->som_y;
    float best = -1.0f;
    int bx = cx, by = cy;

    /* 8-directional neighborhood */
    int dx[] = {-1,-1,-1, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1,-1, 1,-1, 0, 1};

    pthread_rwlock_rdlock(&vm->som.rwlock);
    for (int i = 0; i < 8; i++) {
        int nx = cx + dx[i];
        int ny = cy + dy[i];
        if (nx < 0 || nx >= vm->som.cols) continue;
        if (ny < 0 || ny >= vm->som.rows) continue;
        float act = vm->som.nodes[ny][nx].activation;
        if (act > best) {
            best = act;
            bx = nx;
            by = ny;
        }
    }
    pthread_rwlock_unlock(&vm->som.rwlock);

    a->som_x = bx;
    a->som_y = by;
}

/* SOM_ELECT — the agent with the lowest dist to its BMU becomes leader */
soma_agent_t *soma_elect(soma_vm_t *vm) {
    float best_score = FLT_MAX;
    soma_agent_t *leader = NULL;

    pthread_rwlock_rdlock(&vm->som.rwlock);
    for (int i = 0; i < SOMA_MAX_AGENTS; i++) {
        soma_agent_t *a = &vm->agents[i];
        if (a->id < 0) continue;
        if (atomic_load(&a->state) == SOMA_AGENT_DEAD) continue;

        /* score = distance from agent's R0 to its current SOM node */
        float dist = soma_vec_dist(a->regs.R[0],
                     vm->som.nodes[a->som_y][a->som_x].weights);
        if (dist < best_score) {
            best_score = dist;
            leader = a;
        }
    }
    pthread_rwlock_unlock(&vm->som.rwlock);
    return leader;
}

/* ── Vector ops ──────────────────────────────────────────────────────────── */

float soma_dot(soma_vec_t a, soma_vec_t b) {
    float sum = 0.0f;
    for (int i = 0; i < SOMA_VEC_DIM; i++) sum += a[i] * b[i];
    return sum;
}

/* ── Utility ─────────────────────────────────────────────────────────────── */

void soma_vec_print(const char *label, soma_vec_t v) {
    printf("%s = [", label);
    for (int i = 0; i < SOMA_VEC_DIM; i++) {
        printf("%.4f%s", v[i], i < SOMA_VEC_DIM-1 ? ", " : "");
    }
    printf("]\n");
}

void soma_som_print_activations(soma_vm_t *vm) {
    /* Find max hit count for normalization */
    uint32_t max_hits = 1;
    pthread_rwlock_rdlock(&vm->som.rwlock);
    for (int r = 0; r < vm->som.rows; r++)
        for (int c = 0; c < vm->som.cols; c++)
            if (vm->som.nodes[r][c].hit_count > max_hits)
                max_hits = vm->som.nodes[r][c].hit_count;

    printf("\nSOM Hit Map (%dx%d)  [max hits = %u]\n",
           vm->som.rows, vm->som.cols, max_hits);
    const char *heat = " .,:;+*#@";  /* 9 levels */
    for (int r = 0; r < vm->som.rows; r++) {
        for (int c = 0; c < vm->som.cols; c++) {
            float ratio = (float)vm->som.nodes[r][c].hit_count
                          / (float)max_hits;
            int idx = (int)(ratio * 8.0f);
            if (idx > 8) idx = 8;
            putchar(heat[idx]);
        }
        putchar('\n');
    }
    pthread_rwlock_unlock(&vm->som.rwlock);
}
