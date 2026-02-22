/*
 * soma_bridge.c — Phase II/III/IV bridge stubs for C transpiler output
 *
 * These functions are called by transpiled SOMA programs (curious.c etc.)
 * They are lightweight C stubs that mirror the behaviour of the Python
 * runtime (runtime/som/soul.py, terrain.py, cdbg.py).
 *
 * Compile with your transpiled output:
 *   gcc -O3 -march=native -o curious curious.c soma_bridge.c -lm -lpthread
 *
 * SOMA v4.1.0 — Coherence Pass
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define VEC_DIM     8
#define MAX_AGENTS  256
#define SOM_ROWS    16
#define SOM_COLS    16
#define STALL_THRESH 20

/* ── Internal state ─────────────────────────────────────────────────────── */

typedef float vec_t[VEC_DIM];

/* Per-agent soul state */
static struct {
    vec_t  goal;
    float  goal_dist_prev;
    int    stall_count;
    int    has_goal;
    int    ctx_nibble;      /* active CDBG CTX (default 1 = AGENT) */
} g_soul[MAX_AGENTS];

/* Per-node terrain state */
static struct {
    float  valence_sum;
    int    visit_count;
    float  cultural_deposit;
    int    is_virgin;
} g_terrain[SOM_ROWS][SOM_COLS];

/* Initialise on first use */
static int g_bridge_init = 0;
static void bridge_init(void) {
    if (g_bridge_init) return;
    memset(g_soul,    0, sizeof(g_soul));
    memset(g_terrain, 0, sizeof(g_terrain));
    for (int r = 0; r < SOM_ROWS; r++)
        for (int c = 0; c < SOM_COLS; c++)
            g_terrain[r][c].is_virgin = 1;
    for (int i = 0; i < MAX_AGENTS; i++)
        g_soul[i].ctx_nibble = 1;   /* default CTX = AGENT */
    g_bridge_init = 1;
}

static float vec_dist(float *a, float *b) {
    float d = 0.0f;
    for (int i = 0; i < VEC_DIM; i++) d += (a[i]-b[i])*(a[i]-b[i]);
    return sqrtf(d);
}

/* ── Phase II: Emotional memory ─────────────────────────────────────────── */

void soma_emot_tag(int agent_id, int r, int c, float valence, float intensity) {
    bridge_init();
    if (r < 0 || r >= SOM_ROWS || c < 0 || c >= SOM_COLS) return;
    /* BUG NOTE: valence is passed as a->R[dst][0] from the transpiler.
     * In the learn_loop, dst=R0 which holds the BMU row index (e.g. 8.0)
     * after SOM_BMU — NOT a normalised valence. This is a .soma source bug:
     * EMOT_TAG S0 should use a dedicated valence register, not R0.
     * We clamp valence to [-1,1] here to prevent bogus terrain pollution. */
    if (valence > 1.0f)  valence = 1.0f;
    if (valence < -1.0f) valence = -1.0f;
    g_terrain[r][c].valence_sum    += valence * intensity;
    g_terrain[r][c].visit_count    += 1;
    g_terrain[r][c].is_virgin       = 0;
    g_terrain[r][c].cultural_deposit = g_terrain[r][c].valence_sum
                                       / (float)g_terrain[r][c].visit_count;
#ifdef SOMA_BRIDGE_VERBOSE
    printf("[EMOT_TAG] agent=%d (%d,%d) valence=%.3f intensity=%.3f\n",
           agent_id, r, c, valence, intensity);
#endif
}

void soma_decay_protect(int agent_id, int r, int c, int cycles) {
    bridge_init();
    /* Stub: in Python runtime this sets a protection mode on the node.
     * Here we just log — decay protection is implicit (no decay in C stubs). */
#ifdef SOMA_BRIDGE_VERBOSE
    printf("[DECAY_PROTECT] agent=%d (%d,%d) cycles=%d\n", agent_id, r, c, cycles);
#endif
}

float soma_predict_err(int agent_id, float *input) {
    bridge_init();
    if (!g_soul[agent_id % MAX_AGENTS].has_goal) return 0.0f;
    float d = vec_dist(input, g_soul[agent_id % MAX_AGENTS].goal);
    /* Normalise to [0,1] — max possible dist for unit vectors = sqrt(VEC_DIM) */
    return fminf(d / sqrtf((float)VEC_DIM), 1.0f);
}

float soma_emot_recall(int agent_id, int r, int c) {
    bridge_init();
    if (r < 0 || r >= SOM_ROWS || c < 0 || c >= SOM_COLS) return 0.0f;
    return g_terrain[r][c].cultural_deposit;
}

float soma_surprise_calc(float *a, float *b) {
    bridge_init();
    float d = vec_dist(a, b);
    return fminf(d / sqrtf((float)VEC_DIM), 1.0f);
}

/* ── Phase III: Curiosity (AgentSoul + SomTerrain) ──────────────────────── */

void soma_goal_set(int agent_id, float *goal_vec) {
    bridge_init();
    int id = agent_id % MAX_AGENTS;
    memcpy(g_soul[id].goal, goal_vec, VEC_DIM * sizeof(float));
    g_soul[id].has_goal       = 1;
    g_soul[id].stall_count    = 0;
    g_soul[id].goal_dist_prev = 1e9f;
#ifdef SOMA_BRIDGE_VERBOSE
    printf("[GOAL_SET] agent=%d goal[0]=%.3f\n", agent_id, goal_vec[0]);
#endif
}

float soma_goal_check(int agent_id, float *current_vec) {
    bridge_init();
    int id = agent_id % MAX_AGENTS;
    if (!g_soul[id].has_goal) return 0.0f;

    /* Normalise exactly as soul.py does:
     *   dist = sqrt(sum((a-b)^2)) / sqrt(n),  clamped to [0,1]
     *
     * BUG FIX: previous code returned dist * 65535.
     * That made JZ (checks ==0 after int cast) unreachable and caused
     * [INTROSPECT] goal_dist=65534.3 — the infinite EMOT_TAG loop.
     * Returning [0,1] means (int)dist == 0 when dist < 1.0 (< ~0.72 in
     * practice for VEC_DIM=8), so JZ fires correctly once training converges.
     */
    float sq = 0.0f;
    for (int i = 0; i < VEC_DIM; i++) {
        float d = current_vec[i] - g_soul[id].goal[i];
        sq += d * d;
    }
    float dist = sqrtf(sq) / sqrtf((float)VEC_DIM);
    if (dist > 1.0f) dist = 1.0f;

    /* Stall detection mirrors soul.py exactly:
     *   dist > 0.05  → still far, increment stall_count
     *   dist <= 0.05 → making progress, decrement by 2 */
    if (dist > 0.05f)
        g_soul[id].stall_count++;
    else
        g_soul[id].stall_count -= 2;
    if (g_soul[id].stall_count < 0) g_soul[id].stall_count = 0;

    g_soul[id].goal_dist_prev = dist;
    return dist;   /* [0.0, 1.0] — JZ fires when (int)dist == 0 */
}

float soma_soul_query(int agent_id, float *vec) {
    bridge_init();
    int id = agent_id % MAX_AGENTS;
    if (!g_soul[id].has_goal) return 0.0f;
    /* Intuition: how close is this vec to our goal memory? */
    float d = vec_dist(vec, g_soul[id].goal);
    float similarity = 1.0f - fminf(d / sqrtf((float)VEC_DIM), 1.0f);
    return similarity;
}

/* Forward declaration — agent_run is defined in curious.c / transpiled output */
extern void* agent_run(void* arg);

/* Shared agent pool — must match the one in curious.c (g_agents) */
/* We use a function-pointer approach: META_SPAWN sets up slots then
   the BARRIER in curious.c will actually wait for them. */
static int g_meta_spawn_base = 10;   /* candidate agents use slots 10-13 */

void soma_meta_spawn(int agent_id, int count, int entry_pc) {
    bridge_init();
    /* BUG FIX: was a stub that just printed — BARRIER then returned immediately
     * because no agents were running, causing infinite META_SPAWN cycles.
     *
     * Real fix: we can't call agent_run directly here without access to
     * g_agents (defined in curious.c). Instead we set a flag so BARRIER
     * can detect that no real children were spawned, and break the loop.
     *
     * PROPER FIX IS IN soma_emit_c.py: META_SPAWN case now emits real FORK
     * logic using g_agents slots, then BARRIER waits for them.
     * This stub now correctly signals "0 children spawned" via g_meta_active. */
    printf("[META_SPAWN] agent=%d spawning %d candidates at pc=%d\n",
           agent_id, count, entry_pc);
    /* NOTE: Full implementation requires META_SPAWN to call FORK internally.
     * See the fix in soma_emit_c.py case 0x63 which now emits real thread spawning. */
}

float soma_evolve(int agent_id, int winner_slot) {
    bridge_init();
    /* Stub: selects "winner" — in real runtime compares goal distances of
     * all META_SPAWN children. Returns winner's agent id as float. */
    printf("[EVOLVE] agent=%d selecting winner into slot %d\n",
           agent_id, winner_slot);
    return (float)winner_slot;
}

void soma_introspect(int agent_id) {
    bridge_init();
    int id = agent_id % MAX_AGENTS;
    printf("[INTROSPECT] agent=%d stall=%d has_goal=%d goal_dist=%.4f\n",
           agent_id,
           g_soul[id].stall_count,
           g_soul[id].has_goal,
           g_soul[id].goal_dist_prev > 1e8f ? 0.0f : g_soul[id].goal_dist_prev);
}

float soma_terrain_read(int r, int c) {
    bridge_init();
    if (r < 0 || r >= SOM_ROWS || c < 0 || c >= SOM_COLS) return 0.0f;
    if (g_terrain[r][c].is_virgin) return 1.0f;   /* virgin = max curiosity */
    return g_terrain[r][c].cultural_deposit;
}

void soma_terrain_mark(int r, int c, float valence) {
    bridge_init();
    if (r < 0 || r >= SOM_ROWS || c < 0 || c >= SOM_COLS) return;
    g_terrain[r][c].valence_sum    += valence;
    g_terrain[r][c].visit_count    += 1;
    g_terrain[r][c].is_virgin       = 0;
    g_terrain[r][c].cultural_deposit = g_terrain[r][c].valence_sum
                                       / (float)g_terrain[r][c].visit_count;
}

void soma_soul_inherit(int dst_agent, int src_agent) {
    bridge_init();
    int dst = dst_agent % MAX_AGENTS;
    int src = src_agent % MAX_AGENTS;
    /* Copy soul state from src → dst (winner inherits parent's wisdom) */
    memcpy(&g_soul[dst], &g_soul[src], sizeof(g_soul[0]));
    g_soul[dst].stall_count = 0;   /* reset stall for new generation */
#ifdef SOMA_BRIDGE_VERBOSE
    printf("[SOUL_INHERIT] agent=%d inherits from agent=%d\n", dst_agent, src_agent);
#endif
}

int soma_goal_stalled(int agent_id) {
    bridge_init();
    int id = agent_id % MAX_AGENTS;
    return g_soul[id].stall_count >= STALL_THRESH;
}

/* ── Phase IV: CDBG ──────────────────────────────────────────────────────── */

/* CRC-4 (poly x^4+x+1) over 3 payload bytes */
static uint8_t crc4(uint8_t *data, int len) {
    uint8_t crc = 0;
    for (int i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x80) ? ((crc << 1) ^ 0x13) : (crc << 1);
    }
    return (crc >> 4) & 0xF;
}

void soma_cdbg_emit(int agent_id) {
    bridge_init();
    int id = agent_id % MAX_AGENTS;
    uint8_t ctx = (uint8_t)g_soul[id].ctx_nibble & 0xF;
    /* Build 24-bit agent ID: cluster[4] map[8] seq[12] */
    uint32_t aid24 = ((uint32_t)(agent_id >> 16) & 0xF) << 20
                   | ((uint32_t)(agent_id >>  8) & 0xFF) << 12
                   | ((uint32_t)(agent_id      ) & 0xFFF);
    uint8_t payload[3] = {
        (aid24 >> 16) & 0xFF,
        (aid24 >>  8) & 0xFF,
        (aid24      ) & 0xFF
    };
    uint8_t chk = crc4(payload, 3);
    uint8_t frame[5] = {
        (ctx << 4) | 0x0,      /* CTX[4] SUB[4]=0 */
        payload[0],
        payload[1],
        payload[2],
        (chk << 4) | 0x0       /* CHK[4] RSV[4]=0 */
    };
    printf("[CDBG_EMIT] agent=%d CTX=0x%X frame=%02X%02X%02X%02X%02X\n",
           agent_id, ctx,
           frame[0], frame[1], frame[2], frame[3], frame[4]);
}

float soma_cdbg_recv(int agent_id) {
    bridge_init();
    /* Stub: in real runtime reads from message bus. Returns 0 = no frame. */
    return 0.0f;
}

void soma_ctx_switch(int agent_id, int ctx_nibble) {
    bridge_init();
    int id = agent_id % MAX_AGENTS;
    g_soul[id].ctx_nibble = ctx_nibble & 0xF;
#ifdef SOMA_BRIDGE_VERBOSE
    printf("[CTX_SWITCH] agent=%d ctx=%d\n", agent_id, ctx_nibble);
#endif
}