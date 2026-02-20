/*
 * soma_runtime.h — SOMA Concurrent Agent Runtime
 *
 * True pthreads-based multi-agent runtime.
 * Every agent is a real OS thread with its own message queue.
 * The SOM map is shared, protected by a read-write lock.
 *
 * Architecture:
 *   - Agent     : pthread + register file + message queue + SOM position
 *   - MsgQueue  : lock-free ring buffer (per agent, single producer/consumer)
 *   - SOM       : shared 2D grid of weight vectors (rwlock protected)
 *   - Barrier   : pthread_barrier for FORK/MERGE synchronization
 */

#ifndef SOMA_RUNTIME_H
#define SOMA_RUNTIME_H

#define _GNU_SOURCE
#include <pthread.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <math.h>

/* ── Constants ──────────────────────────────────────────────────────────── */

#define SOMA_MAX_AGENTS     256
#define SOMA_REG_COUNT      16      /* R0–R15  */
#define SOMA_AREG_COUNT     64      /* A0–A63  */
#define SOMA_SREG_COUNT     16      /* S0–S15  */
#define SOMA_VEC_DIM        8       /* 256-bit registers = 8 × f32 */
#define SOMA_MSG_QUEUE_CAP  256     /* ring buffer capacity per agent */
#define SOMA_MAX_SOM_DIM    64      /* max SOM rows or cols */

/* ── Types ──────────────────────────────────────────────────────────────── */

typedef float     soma_vec_t[SOMA_VEC_DIM];   /* 256-bit weight vector  */
typedef uint64_t  soma_instr_t;               /* 64-bit instruction word */
typedef int32_t   soma_agent_id_t;

/* Message envelope */
typedef struct {
    soma_agent_id_t  from;
    soma_vec_t       data;
    uint32_t         tag;
} soma_msg_t;

/* Lock-free ring buffer message queue */
typedef struct {
    soma_msg_t       buf[SOMA_MSG_QUEUE_CAP];
    atomic_int       head;         /* producer writes here */
    atomic_int       tail;         /* consumer reads here  */
    pthread_mutex_t  mu;           /* condvar mutex        */
    pthread_cond_t   not_empty;    /* signal on push       */
    pthread_cond_t   not_full;     /* signal on pop        */
} soma_msgq_t;

/* SOM node — one cell on the topology grid */
typedef struct {
    soma_vec_t  weights;
    float       activation;        /* last BMU activation score */
    uint32_t    hit_count;         /* how many times this node was BMU */
} soma_node_t;

/* SOM map — shared across all agents */
typedef struct {
    soma_node_t      nodes[SOMA_MAX_SOM_DIM][SOMA_MAX_SOM_DIM];
    int              rows;
    int              cols;
    pthread_rwlock_t rwlock;       /* readers: BMU/WALK, writer: TRAIN */
} soma_som_t;

/* Agent register file */
typedef struct {
    soma_vec_t  R[SOMA_REG_COUNT];    /* R0–R15: weight vectors       */
    uint64_t    A[SOMA_AREG_COUNT];   /* A0–A63: agent handles        */
    double      S[SOMA_SREG_COUNT];   /* S0–S15: SOM state            */
                                      /*   S0 = learning rate         */
                                      /*   S1 = sigma (neighborhood)  */
                                      /*   S2 = epoch counter         */
} soma_regs_t;

/* Agent state machine */
typedef enum {
    SOMA_AGENT_READY    = 0,
    SOMA_AGENT_RUNNING  = 1,
    SOMA_AGENT_WAITING  = 2,   /* blocked on MSG_RECV */
    SOMA_AGENT_DEAD     = 3,
} soma_agent_state_t;

/* The agent — one per pthread */
typedef struct soma_agent {
    soma_agent_id_t      id;
    int                  som_x;       /* current position on SOM  */
    int                  som_y;
    soma_regs_t          regs;
    soma_msgq_t          mq;          /* inbound message queue    */
    atomic_int           state;       /* soma_agent_state_t       */
    pthread_t            thread;
    struct soma_vm      *vm;          /* back-pointer to VM       */
    void               (*entry)(struct soma_agent *);  /* thread fn */
} soma_agent_t;

/* Barrier for FORK/MERGE/BARRIER opcodes */
typedef struct {
    pthread_barrier_t  bar;
    int                count;
    bool               initialized;
} soma_barrier_t;

/* The VM — owns everything */
typedef struct soma_vm {
    soma_agent_t    agents[SOMA_MAX_AGENTS];
    int             agent_count;
    soma_som_t      som;
    soma_barrier_t  barriers[32];    /* up to 32 concurrent barriers */
    atomic_int      barrier_count;
    pthread_mutex_t spawn_mu;        /* serialises SPAWN            */
} soma_vm_t;

/* ── API ────────────────────────────────────────────────────────────────── */

/* VM lifecycle */
soma_vm_t  *soma_vm_new(int som_rows, int som_cols);
void        soma_vm_free(soma_vm_t *vm);
void        soma_vm_run(soma_vm_t *vm);       /* blocks until all agents done */

/* Agent management */
soma_agent_t *soma_spawn(soma_vm_t *vm, void (*entry)(soma_agent_t *));
void          soma_agent_kill(soma_agent_t *a);
void          soma_fork(soma_agent_t *parent, int n,
                        void (*entry)(soma_agent_t *));

/* Messaging */
void  soma_msg_send(soma_agent_t *dst, soma_agent_t *src,
                    soma_vec_t data, uint32_t tag);
bool  soma_msg_recv(soma_agent_t *a, soma_msg_t *out);  /* non-blocking */
void  soma_msg_recv_block(soma_agent_t *a, soma_msg_t *out); /* blocking */
void  soma_broadcast(soma_vm_t *vm, soma_agent_t *src,
                     soma_vec_t data, uint32_t tag);

/* Barrier synchronisation */
int   soma_barrier_new(soma_vm_t *vm, int count);
void  soma_barrier_wait(soma_vm_t *vm, int barrier_id);

/* SOM operations */
void  soma_som_init_random(soma_vm_t *vm);
void  soma_som_init_grid(soma_vm_t *vm);
int   soma_bmu(soma_vm_t *vm, soma_vec_t input,
               int *bmu_x, int *bmu_y);        /* returns BMU index */
void  soma_train(soma_vm_t *vm, soma_vec_t input,
                 int bmu_x, int bmu_y,
                 float lr, float sigma);
void  soma_walk_gradient(soma_agent_t *a);     /* move toward activation */
soma_agent_t *soma_elect(soma_vm_t *vm);       /* democratic leader election */
float soma_vec_dist(soma_vec_t a, soma_vec_t b);
void  soma_vec_normalize(soma_vec_t v);

/* SOM state helpers (S registers) */
static inline float soma_lr(soma_agent_t *a)    { return (float)a->regs.S[0]; }
static inline float soma_sigma(soma_agent_t *a) { return (float)a->regs.S[1]; }
static inline void  soma_epoch_inc(soma_agent_t *a) { a->regs.S[2] += 1.0; }

/* Utility */
void soma_vec_print(const char *label, soma_vec_t v);
void soma_som_print_activations(soma_vm_t *vm);

#endif /* SOMA_RUNTIME_H */
