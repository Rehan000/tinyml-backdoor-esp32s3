/*
 * qsentry_bench.cpp -- On-device measurement harness for the ESP32-S3 (ESP-DL).
 *
 * Revision 3 (minor revision 2, Reviewer 4). Two phases, run once at boot:
 *
 *   PHASE 1 -- on-device DETECTION and fidelity, over the FULL embedded test split.
 *     The model is exported with two outputs (logits, gap): the monitor reads the real 64-d pooled
 *     features of each inference, dequantized from INT8 (q * 2^exponent), exactly as deployed. For every
 *     test window the device runs the clean input and, for non-target windows, the triggered input
 *     (x + g_trig, formed on the device), scores both with the class-conditional monitor (CCM), and
 *     computes on the device: accuracy, agreement with host FP32 and with the esp-ppq simulation of the
 *     deployed graph, attack success rate, detection AUROC, TPR at a 1% false-alarm rate, and the
 *     benign alarm rate / detection rate at the calibration-fixed threshold. Per-window records (and,
 *     with DUMP_FEATURES, the raw INT8 feature codes) go to the console for code/ondevice_detection.py.
 *
 *   PHASE 2 -- COST (latency, energy) on the same image. Three regions per round, each preceded by an
 *     idle gap so the PPK2 trace separates them:
 *       A   (GPIO4 / PPK2 D0): inference only                          x N_ITERS_CE
 *       E2E (GPIO6 / PPK2 D2): inference + feature dequant + 1x CCM     x N_ITERS_CE   <- deployed path
 *       B   (GPIO5 / PPK2 D1): inference + N_CCM_REPEAT CCM calls       x N_ITERS_B
 *     (E2E - A) / N_ITERS_CE is the COLD-path per-inference cost of the monitor (its tables and code are
 *     evicted by the inference each time), in both time and energy. B isolates the hot-cache per-call
 *     compute by amplification, as in the previous revision, and is kept only as a lower bound.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "testset.h"
#include "monitor_params.h"

#define MARK_GPIO_INF   GPIO_NUM_4   /* PPK2 D0: region A, inference only                         */
#define MARK_GPIO_DEF   GPIO_NUM_5   /* PPK2 D1: region B, inference + amplified monitor (hot)    */
#define MARK_GPIO_E2E   GPIO_NUM_6   /* PPK2 D2: region E2E, inference + 1x monitor (cold, deployed) */
#define N_WARMUP        20
#define N_ITERS_CE      1000         /* A and E2E: long humps so the cold-path energy delta is measurable */
#define N_ITERS_B       200
#define N_ITERS_DUAL    200
#define N_ROUNDS        12
#define N_CCM_REPEAT    1024
#define DUMP_FEATURES   1            /* print the INT8 feature codes of every window (host bit-check) */

static const char *TAG = "qsentry";

/* Embedded INT8 model (model/model.espdl) and the full test split (testset_x.bin, float32) */
extern const uint8_t model_espdl_start[] asm("_binary_model_espdl_start");
extern const uint8_t testset_x_start[]   asm("_binary_testset_x_bin_start");

static dl::Model *g_model = nullptr;
static dl::TensorBase *g_in = nullptr, *g_logits = nullptr, *g_feat = nullptr;

static inline void mark(gpio_num_t pin, int level) { gpio_set_level(pin, level); }

static void mean_std(const double *v, int n, double *m, double *sd) {
    double s = 0; for (int i = 0; i < n; ++i) s += v[i]; *m = s / n;
    double ss = 0; for (int i = 0; i < n; ++i) { double d = v[i] - *m; ss += d * d; }
    *sd = (n > 1) ? sqrt(ss / (n - 1)) : 0.0;
}

static int tensor_numel(dl::TensorBase *t) {
    int n = 1;
    for (int d : t->shape) n *= d;
    return n;
}

static inline int elem_int(dl::TensorBase *t, int i) {
    return (t->dtype == dl::DATA_TYPE_INT16) ? ((int16_t *)t->data)[i] : ((int8_t *)t->data)[i];
}

/* Bind the input and the two outputs (by size: N_CLASSES logits, MONITOR_DIM features). */
static bool bind_tensors(void) {
    g_in = g_model->get_inputs().begin()->second;
    for (auto &kv : g_model->get_outputs()) {
        int n = tensor_numel(kv.second);
        ESP_LOGI(TAG, "output '%s': numel=%d dtype=%s exponent=%d", kv.first.c_str(), n,
                 dl::dtype_to_string(kv.second->dtype), (int)kv.second->exponent);
        if (n == N_CLASSES) g_logits = kv.second;
        else if (n == MONITOR_DIM) g_feat = kv.second;
    }
    if (!g_logits || !g_feat) {
        ESP_LOGE(TAG, "model must expose logits[%d] and features[%d]: re-export with "
                      "convert_espdl.py --with-features", N_CLASSES, MONITOR_DIM);
        return false;
    }
    return true;
}

/* Quantize a float signal into the model input and run one INT8 inference; return argmax. */
static int model_infer(const float *signal, dl::runtime_mode_t mode = dl::RUNTIME_MODE_SINGLE_CORE) {
    const float inv_scale = 1.0f / exp2f((int)g_in->exponent);   /* real = q * 2^exponent */
    int8_t *q = (int8_t *)g_in->data;
    for (int i = 0; i < SIG_LEN; ++i) {
        int v = (int)lroundf(signal[i] * inv_scale);
        q[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }
    g_model->run(mode);
    int best = 0, bv = elem_int(g_logits, 0);                     /* argmax is scale-invariant */
    for (int i = 1; i < N_CLASSES; ++i) { int v = elem_int(g_logits, i); if (v > bv) { bv = v; best = i; } }
    return best;
}

/* Dequantize the resident pooled features of the last inference (the monitor's input). */
static inline void read_features(float *gap) {
    const float s = exp2f((int)g_feat->exponent);
    for (int d = 0; d < MONITOR_DIM; ++d) gap[d] = (float)elem_int(g_feat, d) * s;
}

/* CCM: class-conditional robust activation score (matches tinybd.defense._score_classcond) */
static float ccm_score(const float *gap, int pred_class) {
    const float *md = g_median[pred_class];
    const float *iv = g_inv_mad[pred_class];
    float acc = 0.0f;
    for (int d = 0; d < MONITOR_DIM; ++d) {
        float z = fabsf(gap[d] - md[d]) * iv[d];
        if (z > CCM_CLIP_K) z = CCM_CLIP_K;
        acc += z;
    }
    return acc / (float)MONITOR_DIM;
}

/* ---------------- on-device detection metrics (same definitions as tinybd.metrics) ------------- */
static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y);
}

/* rank-free AUROC: P(pos > neg) + 0.5 P(pos == neg), exact over all pairs */
static double auroc(const float *pos, int np_, const float *neg, int nn) {
    double s = 0;
    for (int i = 0; i < np_; ++i)
        for (int j = 0; j < nn; ++j) s += (pos[i] > neg[j]) ? 1.0 : (pos[i] == neg[j] ? 0.5 : 0.0);
    return s / ((double)np_ * nn);
}

/* TPR at a threshold giving <= fpr on the negatives (tinybd.metrics.tpr_at_fpr) */
static double tpr_at_fpr(const float *pos, int np_, const float *neg, int nn, double fpr, float *scratch) {
    memcpy(scratch, neg, sizeof(float) * nn);
    qsort(scratch, nn, sizeof(float), cmp_float);
    int k = (int)ceil((1.0 - fpr) * nn) - 1; if (k < 0) k = 0;
    float thr = scratch[k]; int hit = 0;
    for (int i = 0; i < np_; ++i) hit += pos[i] > thr;
    return (double)hit / np_;
}

static void print_features(char tag, int i) {
    printf("F,%c,%d", tag, i);
    for (int d = 0; d < MONITOR_DIM; ++d) printf(",%d", elem_int(g_feat, d));
    printf("\n");
}

static float s_x[SIG_LEN];                  /* aligned working copy of one window */
static float s_clean[N_TEST], s_trig[N_TEST], s_negnt[N_TEST], s_scratch[N_TEST];

static void phase_detection(void) {
    const float *X = (const float *)testset_x_start;   /* may be unaligned: copy with memcpy */
    float gap[MONITOR_DIM];
    int correct = 0, agree_fp = 0, agree_sim = 0, n_sim = 0, n_nt = 0, asr = 0, n_neg_nt = 0;
    int far_all = 0, far_nt = 0, det = 0;
    int64_t t0 = esp_timer_get_time();
    printf("BEGIN_DETECTION,N_TEST=%d,TARGET=%d,THR=%.8e\n", N_TEST, TARGET_CLASS, (double)CCM_THRESHOLD);
    for (int i = 0; i < N_TEST; ++i) {
        memcpy(s_x, X + (size_t)i * SIG_LEN, sizeof(s_x));
        int p = model_infer(s_x);
        read_features(gap);
        float sc = ccm_score(gap, p);
        s_clean[i] = sc;
        if (DUMP_FEATURES) print_features('c', i);
        printf("D,%d,%d,0,%d,%.8e\n", i, g_test_y[i], p, (double)sc);
        correct += (p == g_test_y[i]);
        agree_fp += (p == g_test_pred_fp[i]);
        if (g_test_pred_sim[i] >= 0) { agree_sim += (p == g_test_pred_sim[i]); n_sim++; }
        far_all += (sc > CCM_THRESHOLD);
        if (g_test_y[i] != TARGET_CLASS) {
            s_negnt[n_neg_nt++] = sc; far_nt += (sc > CCM_THRESHOLD);
            for (int t = 0; t < SIG_LEN; ++t) s_x[t] += g_trig[t];
            int pt = model_infer(s_x);
            read_features(gap);
            float st = ccm_score(gap, pt);
            s_trig[n_nt++] = st;
            if (DUMP_FEATURES) print_features('t', i);
            printf("D,%d,%d,1,%d,%.8e\n", i, g_test_y[i], pt, (double)st);
            asr += (pt == TARGET_CLASS);
            det += (st > CCM_THRESHOLD);
        }
    }
    double secs = (esp_timer_get_time() - t0) / 1e6;
    /* paper protocol (Def.evaluate): negatives = clean NON-target windows, positives = the same triggered */
    double au = auroc(s_trig, n_nt, s_negnt, n_neg_nt);
    double t1 = tpr_at_fpr(s_trig, n_nt, s_negnt, n_neg_nt, 0.01, s_scratch);
    double t5 = tpr_at_fpr(s_trig, n_nt, s_negnt, n_neg_nt, 0.05, s_scratch);
    /* stricter: every clean window (all classes) as a negative */
    double au_all = auroc(s_trig, n_nt, s_clean, N_TEST);
    double t1_all = tpr_at_fpr(s_trig, n_nt, s_clean, N_TEST, 0.01, s_scratch);
    printf("END_DETECTION\n");
    printf("SUMMARY,acc=%.4f,agree_fp32=%.4f,agree_sim=%.4f,asr=%.4f,auroc=%.4f,tpr1=%.4f,tpr5=%.4f,"
           "auroc_allneg=%.4f,tpr1_allneg=%.4f,far_cal_all=%.4f,far_cal_nontarget=%.4f,tpr_cal=%.4f,"
           "n_test=%d,n_trig=%d,secs=%.1f\n",
           (double)correct / N_TEST, (double)agree_fp / N_TEST, n_sim ? (double)agree_sim / n_sim : -1.0,
           (double)asr / n_nt, au, t1, t5, au_all, t1_all, (double)far_all / N_TEST,
           (double)far_nt / n_neg_nt, (double)det / n_nt, N_TEST, n_nt, secs);
    ESP_LOGI(TAG, "on-device: acc=%.3f agree(FP32)=%.3f agree(sim)=%.3f ASR=%.3f AUROC=%.4f TPR@1%%=%.3f "
             "FAR@cal-thr=%.4f TPR@cal-thr=%.3f (N=%d clean, %d triggered)",
             (double)correct / N_TEST, (double)agree_fp / N_TEST, n_sim ? (double)agree_sim / n_sim : -1.0,
             (double)asr / n_nt, au, t1, (double)far_all / N_TEST, (double)det / n_nt, N_TEST, n_nt);
}

static void phase_cost(void) {
    float gap[MONITOR_DIM];
    memcpy(s_x, testset_x_start, sizeof(s_x));     /* a real CWRU window (compute is data-independent) */

    int64_t td0 = esp_timer_get_time();
    for (int i = 0; i < N_ITERS_DUAL; ++i) model_infer(s_x, dl::RUNTIME_MODE_MULTI_CORE);
    ESP_LOGI(TAG, "dual-core latency: %.2f us/inf", (double)(esp_timer_get_time() - td0) / N_ITERS_DUAL);

    vTaskDelay(pdMS_TO_TICKS(5000));   /* long idle: marks the start of the energy rounds in the trace */
    printf("BEGIN_COST,rounds=%d,iters_ce=%d,iters_b=%d,repeat=%d\n", N_ROUNDS, N_ITERS_CE, N_ITERS_B,
           N_CCM_REPEAT);

    float acc = 0.0f; volatile float sink = 0.0f; int cc = 0, pred = 0;
    static double infer_us[N_ROUNDS], e2e_us[N_ROUNDS], perccm_us[N_ROUNDS];
    for (int round = 0; round < N_ROUNDS; ++round) {
        vTaskDelay(pdMS_TO_TICKS(500));

        /* Region A: inference only */
        int64_t t0 = esp_timer_get_time();
        mark(MARK_GPIO_INF, 1);
        for (int i = 0; i < N_ITERS_CE; ++i) pred = model_infer(s_x);
        mark(MARK_GPIO_INF, 0);
        int64_t t_inf = esp_timer_get_time() - t0;

        vTaskDelay(pdMS_TO_TICKS(800));

        /* Region E2E: the DEPLOYED path -- inference, read the resident features, one monitor call */
        t0 = esp_timer_get_time();
        mark(MARK_GPIO_E2E, 1);
        for (int i = 0; i < N_ITERS_CE; ++i) {
            pred = model_infer(s_x);
            read_features(gap);
            acc += ccm_score(gap, pred);
        }
        mark(MARK_GPIO_E2E, 0);
        int64_t t_e2e = esp_timer_get_time() - t0;

        vTaskDelay(pdMS_TO_TICKS(800));

        /* Region B: inference + amplified monitor (hot cache) -> per-call compute lower bound */
        t0 = esp_timer_get_time();
        mark(MARK_GPIO_DEF, 1);
        for (int i = 0; i < N_ITERS_B; ++i) {
            pred = model_infer(s_x);
            read_features(gap);
            for (int r = 0; r < N_CCM_REPEAT; ++r) acc += ccm_score(gap, (cc++) % N_CLASSES);
        }
        mark(MARK_GPIO_DEF, 0);
        int64_t t_b = esp_timer_get_time() - t0;
        sink = acc;

        infer_us[round]  = (double)t_inf / N_ITERS_CE;
        e2e_us[round]    = (double)t_e2e / N_ITERS_CE;
        perccm_us[round] = ((double)t_b / N_ITERS_B - infer_us[round]) / N_CCM_REPEAT;
        ESP_LOGI(TAG, "[round %2d] infer %.2f us | end-to-end %.2f us | per-CCM (hot) %.3f us",
                 round, infer_us[round], e2e_us[round], perccm_us[round]);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    double mi, si, me, se, mc, sc2;
    mean_std(infer_us, N_ROUNDS, &mi, &si);
    mean_std(e2e_us, N_ROUNDS, &me, &se);
    mean_std(perccm_us, N_ROUNDS, &mc, &sc2);
    ESP_LOGI(TAG, "==== COST SUMMARY (single-core, %d rounds; A/E2E %d iters, B %d iters) ====",
             N_ROUNDS, N_ITERS_CE, N_ITERS_B);
    ESP_LOGI(TAG, "inference     : %.1f +/- %.1f us/inf", mi, si);
    ESP_LOGI(TAG, "end-to-end    : %.1f +/- %.1f us/inf  (inference + input-quant + feature read + 1x CCM)", me, se);
    ESP_LOGI(TAG, "CCM overhead  : %.2f us (%.2f%%) cold  [hot per-call: %.3f +/- %.3f us]",
             me - mi, 100.0 * (me - mi) / mi, mc, sc2);
    printf("COST,infer_us=%.2f,infer_sd=%.2f,e2e_us=%.2f,e2e_sd=%.2f,ccm_cold_us=%.2f,ccm_hot_us=%.3f\n",
           mi, si, me, se, me - mi, mc);
    printf("END_COST,sink=%.0f\n", (double)sink);
}

extern "C" void app_main(void) {
    const gpio_num_t pins[] = {MARK_GPIO_INF, MARK_GPIO_DEF, MARK_GPIO_E2E};
    for (gpio_num_t p : pins) { gpio_reset_pin(p); gpio_set_direction(p, GPIO_MODE_OUTPUT); gpio_set_level(p, 0); }

    size_t int_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    g_model = new dl::Model((const char *)model_espdl_start, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    size_t int_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t psram_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "model RAM: internal=%d B  psram=%d B",
             (int)(int_before - int_after), (int)(psram_before - psram_after));
    ESP_LOGI(TAG, "config: CPU=%d MHz  PSRAM total=%d KB free=%d KB  internal free=%d KB  N_TEST=%d",
             (int)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
             (int)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024),
             (int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
             (int)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024), N_TEST);
    printf("RAM,internal=%d,psram=%d\n", (int)(int_before - int_after), (int)(psram_before - psram_after));
    if (!bind_tensors()) { while (1) vTaskDelay(pdMS_TO_TICKS(5000)); }

    memcpy(s_x, testset_x_start, sizeof(s_x));
    for (int i = 0; i < N_WARMUP; ++i) model_infer(s_x);

    phase_detection();
    phase_cost();
    printf("DONE\n");
    while (1) vTaskDelay(pdMS_TO_TICKS(5000));
}
