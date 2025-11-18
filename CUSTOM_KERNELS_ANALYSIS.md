# Custom Kernels and Attention Mechanisms Analysis for Causal Language Models

**QNN (Qualcomm AI Engine Direct) Version:** 2.40.0.251030
**Analysis Date:** 2025-11-18
**Target Platform:** Qualcomm Snapdragon (HTP/GPU)

---

## Executive Summary

This document provides a comprehensive analysis of custom kernels, attention mechanisms, and intrinsic code implementations in the QNN codebase that are optimized for causal language models (LLMs) on Qualcomm hardware platforms.

**Key Findings:**
- ✅ Production-ready causal attention implementation with multiple modes
- ✅ Hardware-optimized KV cache management (HMX native layout)
- ✅ Extensive HVX (Hexagon Vector Extensions) intrinsics library
- ✅ Quantized inference support (8-bit, 16-bit)
- ✅ Multi-variant graph system for efficient prefill/decode
- ✅ Advanced context management for long-context LLMs

---

## Table of Contents

1. [Attention Mechanisms](#1-attention-mechanisms)
2. [KV Cache Management](#2-kv-cache-management)
3. [Transformer Architecture](#3-transformer-architecture)
4. [HVX Intrinsics](#4-hvx-intrinsics)
5. [Performance Optimizations](#5-performance-optimizations)
6. [Supported Models](#6-supported-models)
7. [Code Examples](#7-code-examples)

---

## 1. Attention Mechanisms

### 1.1 Causal Attention Implementation

**Primary Files:**
- `examples/Genie/Genie/src/qualla/engines/qnn-htp/attention-mask.hpp` (130 lines)
- `examples/Genie/Genie/src/qualla/engines/qnn-htp/attention-mask.cpp` (233 lines)

**Three Attention Modes:**

| Mode | Purpose | Use Case |
|------|---------|----------|
| **CAUSAL** | Full causal attention with contiguous spans | Standard autoregressive LLMs |
| **RELATIONAL** | Tree-based sparse attention | Structured generation patterns |
| **CUSTOM** | Fully-specified 2D attention masks | Custom attention patterns |

**Core Causal Logic** (attention-mask.cpp:72-79):
```cpp
// Full causal attention - single contiguous span
if (past_idx + n_valid_kv == new_idx) {
  spans.emplace_back(past_idx, n_valid_kv + query_token_idx + 1);
} else {
  spans.emplace_back(past_idx, n_valid_kv);
  spans.emplace_back(new_idx, query_token_idx + 1);
}
```

**Key Features:**
- **AttentionSpan Structure**: Represents contiguous attention regions for batch operations
- **Vectorized Mask Filling**: Uses `PRAGMA_LOOP_VECTORIZE` for SIMD optimization (lines 129, 137, 163)
- **Variable Bitwidth Support**: Template-based for uint8_t, uint16_t, uint32_t
- **Position ID Generation**: RoPE (Rotary Position Embeddings) support (lines 154-181)
- **SSD Prefix Skipping**: Optimization for Speculative Sampling Decode (lines 183-207)

**Performance Optimizations:**
- Loop vectorization with SIMD intrinsics
- Attention span merging to reduce iterations
- Lazy position ID calculation for relational masks

---

## 2. KV Cache Management

### 2.1 KVManager Architecture

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-htp/KVCache/kvmanager.hpp` (461 lines)

**InferenceStep Structure** (lines 36-66):
```cpp
struct InferenceStep {
  uint32_t variant;      // AR (Attention Ratio) - tokens per step
  uint32_t ctx_size;     // CL (Context Length) - max sequence length
  uint32_t n_past;       // Total tokens processed
  uint32_t n_valid_kv;   // Valid KV entries
  uint32_t n_process;    // Tokens to process in this step
  uint32_t past_idx;     // KV cache past index
  uint32_t new_idx;      // KV cache new index
};
```

**KV Update Methods** (nsp-model.cpp:639-645):
```cpp
enum class KVManagerMode {
  SHIFT_CONCAT  = 0x1,  // Deprecated
  SMART_MASK    = 0x2,  // Intelligent masking (default)
  NATIVE_KV     = 0x3,  // HMX native layout for hardware acceleration
  POINTER_SHIFT = 0x4   // Deprecated
};
```

### 2.2 Context Management Strategies

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-htp/KVCache/context-manager.hpp` (128 lines)

**Three Strategies:**

1. **SlidingWindow** (lines 61-81):
   - Maintains recent token window for long contexts
   - Efficient memory usage for extended generations
   - Queue-based tracking: `recent_idxes`

2. **KeyDiff** (lines 83-126):
   - Advanced eviction algorithm based on key similarity
   - Runs scoring model to determine important tokens
   - Methods:
     - `runScorer()`: Execute scoring model on HTP
     - `updateEvictionIndexes()`: Determine which tokens to evict
     - `clearAnchor()`, `updateAnchor()`: Manage scoring buffer state
   - Optimized for long-context language models

3. **Base ContextManager** (lines 24-59):
   - `processUpdate()`: Handle KV cache updates
   - `processMove()`: Switch between graph variants
   - `processReduce()`: Remove entries from cache
   - `translateAttentionMask()`: Global to group attention translation

### 2.3 Native KV Implementation (Hardware Accelerated)

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-htp/KVCache/native-kv.hpp` (94 lines)

**Hardware-Optimized Layout:**
```cpp
static const uint32_t K_TILE = 256;    // Key cache tile size
static const uint32_t V_TILE = 64;     // Value cache tile size
static const uint32_t KV_BLOCK_SIZE = 1024;  // Total block
```

**Features:**
- Optimized for HMX (Hexagon Matrix Extensions) operations
- Tile-based layout for vectorized operations
- Automatic detection when `QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT` is used
- Eliminates format conversion overhead

### 2.4 SmartMask KV Manager

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-htp/KVCache/smart-mask.hpp` (93 lines)

**Key Methods:**
- `completeInit()`: Initialize smart masking patterns
- `getIndexForNewKV()`: Calculate optimal KV insertion index
- `updateKV()`, `reduceKV()`, `moveKV()`: Core KV operations
- `reshapeCache()`: Dynamic cache resizing for variant switches
- `dumpCache()`/`loadCache()`: Persistent cache serialization

---

## 3. Transformer Architecture

### 3.1 Graph Types for Autoregressive Generation

**File:** `examples/Genie/Genie/src/AUTOREGRESSIVE_TEXT_GENERATION.md` (1216 lines)

**Enum Classification** (lines 101-109):
```cpp
enum class GraphType {
  NONE,            // Unclassified
  DEFAULT,         // Generic graph
  LUT,             // Lookup table (embedding layer only)
  DECODER,         // Transformer decoder with hidden state outputs
  DECODER_PREFILL, // Decoder optimized for prefill (KV cache outputs only)
  LMHEAD,          // Language model head (final projection to vocabulary)
  IMAGE_ENCODER    // Vision encoder for multimodal models
};
```

**Variant Naming Convention** (lines 57-90):
- **AR-X_CL-Y** format:
  - **AR (Attention Ratio)**: Number of tokens per step (1, 8, 64, etc.)
  - **CL (Context Length)**: Maximum sequence length (2048, 4096, 8192, etc.)
- Example: `llama_AR-64_CL-4096_decoder`
  - Processes 64 tokens per inference step
  - Supports up to 4096 tokens in context

### 3.2 Model Configuration

**Files:**
- `include/QNN/GenAiTransformer/QnnGenAiTransformerCommon.h` (50 lines)
- Configuration examples: `examples/Genie/configs/llama2-7b/llama2-7b-genaitransformer.json`

**Backend Identifier:**
```cpp
#define QNN_BACKEND_ID_GENAI_TRANSFORMER 14
#define QNN_GENAI_TRANSFORMER_INTERFACE_PROVIDER_NAME "GENAI_TRANSFORMER_QTI_AISW"
```

### 3.3 Two-Phase Generation Pipeline

**Phase 1: Prefill** (Lines 368-382):
- Input: N prompt tokens (e.g., 50)
- Select AR-64 variant for batch processing
- Initialize KV cache with all tokens
- Output: Logits for last token only
- Performance: High throughput (tokens/sec)

**Phase 2: Decode Loop** (Lines 386-414):
- Input: 1 generated token per step
- Select AR-1 variant for single-token
- Attend to all past tokens via causal mask
- Update KV cache incrementally
- Output: Logits for current position
- Performance: Lower throughput, minimal latency

---

## 4. HVX Intrinsics

### 4.1 Core Intrinsics Library

**File:** `include/QNN/HTP/core/intrinsics.h` (826 lines)

#### 4.1.1 Vector Load/Store Operations

**Unaligned Memory Support** (Lines 158-199):
```cpp
inline HVX_Vector q6op_V_vldu_A(void const *addr)
    // Unaligned vector load (128 bytes)

inline void q6op_vstu_AV(void *addr, HVX_Vector v)
    // Unaligned vector store

inline void q6op_vstu_QAV(HVX_VectorPred Qmask, void *addr, HVX_Vector v)
    // Masked unaligned store (for boundary conditions)

inline void q6op_vstu_variable_ARV(void *addr, int n, HVX_Vector vin)
    // Store first n bytes (1..128) - handles irregular tensor boundaries
```

**Use Case:** Load/store attention matrices (Q, K, V) that may not be 128-byte aligned.

#### 4.1.2 Cache Prefetching

**Critical for Attention Performance** (Lines 299-350):
```cpp
inline void dcfetch(void const *addr)
    { Q6_dcfetch_A(addr); }

inline void dcfetch_multi(void const *addr, int len)
    { asm volatile(" dcfetch_multi(%0,%1) " : : "r"(addr), "r"(len)); }

inline void l2pref(const void *p, uint32_t height, uint32_t width, uint32_t stride)
{
    uint32_t control = (height << 16) | (width << 11) | stride;
    asm volatile(" l2fetch(%0,%1) " : : "r"(p), "r"(control));
}

inline void pause_just_enough()
    { asm volatile("pause(#255)"); }  // Pipeline control

inline void unpause()
    { asm volatile("unpause"); }
```

**Use Case:**
- `dcfetch_multi`: Prefetch blocks of attention data
- `l2fetch`: Prefetch 2D matrices with stride patterns (perfect for Q×K^T)
- `pause/unpause`: Control DSP pipeline for optimal throughput

#### 4.1.3 64-bit Accumulation

**For Attention Score Reduction** (Lines 504-534):
```cpp
inline HVX_VectorPair addv_u64(HVX_VectorPair acc, HVX_Vector newdata)
{
    // Add 32-bit data to 64-bit accumulator without overflow
    HVX_Vector newlo = newdata;
    HVX_Vector newhi = Q6_V_vzero();
    HVX_Vector acc_lo = Q6_V_lo_W(acc);
    HVX_Vector acc_hi = Q6_V_hi_W(acc);

    HVX_Vector sum_lo = Q6_Vw_vadd_VwVw(acc_lo, newlo);
    HVX_VectorPred overflow = Q6_Q_vcmp_gt_VuwVuw(newlo, sum_lo);
    HVX_Vector sum_hi = Q6_Vw_condacc_QVwVw(overflow, acc_hi, Q6_V_vsplat_R(1));
    sum_hi = Q6_Vw_vadd_VwVw(sum_hi, newhi);

    return Q6_W_vcombine_VV(sum_hi, sum_lo);
}
```

**Use Case:** Accumulate attention scores across all tokens without integer overflow.

#### 4.1.4 Float Conversion

**For Softmax in Attention** (Lines 539-680):
```cpp
inline HVX_Vector uint64_to_qfloat(HVX_Vector ll_hi, HVX_Vector ll_lo)
{
    // Convert 64-bit integer to Qualcomm qf32 format
    // 1. Find MSB position (normalization)
    // 2. Extract mantissa (top 24 bits)
    // 3. Compute exponent (127 + MSB position)
    // 4. Assemble: (exponent << 23) | mantissa
}

inline HVX_Vector uint64_to_float(HVX_VectorPair bigval)
{
    // Convert to IEEE 754 float32
}

static inline HVX_Vector Q6_Vqf32_equals_Vsf(HVX_Vector vin)
    { return Q6_Vqf32_vadd_VsfVsf(vin, Q6_V_vzero()); }

static inline HVX_Vector int32_to_fp32(HVX_Vector vin)
    // Sign extraction, MSB position, normalization
```

**Use Case:** Convert accumulated attention scores to float for softmax normalization.

### 4.2 HVX Math Operations

**File:** `include/QNN/HTP/core/hvx_mathops.h` (85 lines)

**Float16 to Int16 Conversion** (Lines 38-80):
```cpp
template <int FBITS, bool RND>
inline HVX_Vector s16_from_hf_core(HVX_Vector vin)
{
    // Scale by 2^FBITS for fixed-point conversion
    HVX_VectorPair v32 = Q6_Wqf32_vmpy_VhfVhf(
        vin,
        Q6_Vh_vsplat_R(0x3C00 + FBITS * 0x400)  // Scale factor
    );

    // Add 192K bias for exponent extraction
    HVX_Vector v192K = Q6_V_vsplat_R(0x48400000);
    HVX_Vector vsf_0 = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(v32), v192K)
    );

    // Round to 16-bit with saturation
    if constexpr (RND) {
        result = Q6_Vh_vround_VwVw_sat(vsf_1, vsf_0);
    }

    // Handle overflow with lookup table
    HVX_Vector saturated = Q6_Vh_vlut4_VuhPh(vin, 0x800080007fff7fffULL);
}
```

**Key Intrinsics:**
- `Q6_Wqf32_vmpy_VhfVhf`: Float16 multiply → 64-bit QF32 pair
- `Q6_Vh_vround_VwVw_sat`: 32-bit → 16-bit rounding with saturation
- `Q6_Vh_vlut4_VuhPh`: 4-bit lookup table (saturation limits)

**Use Case:** Quantize attention scores from float16 to int16 for efficient storage.

### 4.3 Compiler Builtin Intrinsics

**File:** `include/QNN/HTP/core/builtin_intrinsics.h` (248 lines)

**Branch Prediction:**
```cpp
#define HEX_LIKELY(x)   __builtin_expect(!!(x), 1)
#define HEX_UNLIKELY(x) __builtin_expect(!!(x), 0)
```

**Atomic Operations:**
```cpp
#define HEX_ATOMIC_FETCH_AND_ADD   __sync_fetch_and_add
#define HEX_ATOMIC_BOOL_COMPARE_AND_SWAP __sync_bool_compare_and_swap
```

**Bit Manipulation:**
```cpp
#define HEX_COUNT_LEADING_ZERO  __builtin_clz   // MSB position for normalization
#define HEX_COUNT_TRAILING_ZERO __builtin_ctz
#define HEX_POPCOUNT            __builtin_popcount
```

### 4.4 Example Custom Kernels

#### 4.4.1 HVX ReLU

**File:** `examples/QNN/OpPackage/HTP/ExampleOpPackageRelu.cpp` (438 lines)

**8-bit Quantized ReLU** (Lines 315-323):
```cpp
HVX_Vector vOmin = Q6_Vb_vsplat_R(minClip);  // Splat min value to 128 bytes
HVX_Vector vOmax = Q6_Vb_vsplat_R(maxClip);  // Splat max value

for (uint32_t i = 0; i < inBlocks; ++i) {
    auto in_vptr  = (const HVX_Vector *)(inBlockTab[i]);
    auto out_vptr = (HVX_Vector *)(outBlockTab[i]);

    for (uint32_t j = 0; j < 16; ++j) {
        out_vptr[j] = Q6_Vub_vmin_VubVub(
            Q6_Vub_vmax_VubVub(in_vptr[j], vOmin),  // max(x, min_clip)
            vOmax                                     // min(result, max_clip)
        );
    }
}
```

#### 4.4.2 HVX MaxPool

**File:** `examples/QNN/OpPackage/HTP/ExampleOpPackageMaxPool.cpp` (400+ lines)

**Vertical Max Reduction** (Lines 183-188):
```cpp
inline HVX_VectorPair verticalMax3(
    HVX_Vector *iptr0, HVX_Vector *iptr1, HVX_Vector *iptr2)
{
    HVX_Vector maxT = Q6_Vub_vmax_VubVub(iptr1[0], iptr1[2]);
    HVX_Vector max0 = Q6_Vub_vmax_VubVub(maxT, iptr0[0]);
    HVX_Vector max1 = Q6_Vub_vmax_VubVub(maxT, iptr2[0]);
    return Q6_W_vcombine_VV(max1, max0);
}
```

**Horizontal Max with Deinterleaving** (Lines 279-298):
```cpp
// Deinterleave odd-even elements
HVX_VectorPair W_Odds_Evens0 = Q6_W_vdeal_VVR(max1lo, max0lo, -32);
HVX_VectorPair W_Odds_Evens1 = Q6_W_vdeal_VVR(max1hi, max0hi, -32);

HVX_Vector out0 = Q6_Vub_vmax_VubVub(
    Q6_V_hi_W(W_Odds_Evens0),
    Q6_V_lo_W(W_Odds_Evens0)
);
```

---

## 5. Performance Optimizations

### 5.1 Inference Strategy Optimization

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-htp/KVCache/kvmanager.cpp` (1160 lines)

**Dynamic Graph Selection** (Lines 360-461):
1. **Prefill**: Select large AR variant (AR-64) for batch processing
2. **Decode**: Select AR-1 for single-token generation
3. **Adaptive**: Choose optimal variant based on remaining tokens

**Context Size Management:**
- Lower bound search for appropriate context size
- Dynamic context expansion when needed
- Prevents unnecessary memory allocation

### 5.2 LMHEAD Skipping Optimization

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-htp/nsp-model.cpp` (Lines 1809-1812)

```cpp
// Skip LMHEAD graph if:
// 1. Not requesting all outputs (output_all=false)
// 2. Not the final inference step
if (nsp_graph.m_graphType == GraphType::LMHEAD &&
    (!output_all) &&
    !m_kvmanager->isFinalInferenceStep()) {
  continue;  // Skip LMHEAD execution
}
```

**Performance Benefit:** For prefill with 256 tokens as 4×AR-64, saves 3× computation by only computing logits for final step.

### 5.3 Precomputed RoPE Embeddings

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-htp/nsp-model.cpp` (Lines 2291-2413)

**Optimization:**
- Pre-compute sin/cos tables for all positions: `[ctx_size, pos_dim]`
- Lookup during inference (O(1)) instead of trigonometry (O(n))
- Supports RoPE scaling (LLaMA3, LongRope variants)
- Quantization-aware computation

**Implementation:**
```cpp
// Line 125-128: RoPE embeddings
uint32_t m_pos_dim{0};
void* rope_sin{nullptr};  // [ctx_size, m_pos_dim]
void* rope_cos{nullptr};  // [ctx_size, m_pos_dim]
```

### 5.4 Hardware-Optimized KV Layout

**Detection** (Lines 348-357 in nsp-model.cpp):
```cpp
if (tspec.tensor->v1.dataFormat == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT) {
  _kv_update_method = KVManagerMode::NATIVE_KV;
}
```

**Benefits:**
- Uses hardware-native KV cache format
- Optimized for HMX (Hexagon Matrix Extensions)
- Eliminates format conversion overhead

### 5.5 Vectorization Pragmas

**Attention Mask Filling** (attention-mask.cpp):
```cpp
PRAGMA_LOOP_VECTORIZE
for (size_t j = 0; j < n_valid_kv; j++) {
    if (attention_row[n_past - n_valid_kv + j]) {
        attention_buffer[past_idx + j] = pos_val;
    }
}
```

**Compiler Transformation:** Automatically converted to HVX vector operations (128-byte SIMD).

---

## 6. Supported Models

### 6.1 Model Zoo

**Configuration Directory:** `examples/Genie/configs/`

| Model | Variant | Parameters | Configuration File |
|-------|---------|------------|-------------------|
| **Llama 2** | 7B | 7 billion | `llama2-7b/llama2-7b-genaitransformer.json` |
| **Llama 3** | 8B-Instruct | 8 billion | `llama3-8b/llama3-8b-genaitransformer.json` |
| **Phi 3** | Mini | 3.8 billion | `phi3-mini/phi3-mini-genaitransformer.json` |
| **LLaVA** | Multimodal | Vision+Language | `llava/llava-genaitransformer.json` |
| **BGE** | Embeddings | Text embeddings | `bge/bge-genaitransformer.json` |

### 6.2 Supported Operations

**File:** `include/QNN/QnnOpDef.h` (800+ lines)

**Key Transformer Operations:**

| Operation | QNN Op Name | Purpose | Line |
|-----------|-------------|---------|------|
| **MatMul** | `QNN_OP_MAT_MUL` | Q-K and Value projections | 522 |
| **MaskedSoftmax** | `QNN_OP_MASKED_SOFTMAX` | Causal softmax with masking | 482 |
| **Softmax** | `QNN_OP_SOFTMAX` | Attention weight normalization | 674 |
| **LayerNorm** | `QNN_OP_LAYER_NORM` | Pre/post-layer normalization | 449 |
| **LogSoftmax** | `QNN_OP_LOG_SOFTMAX` | Log-space softmax for stability | 455 |

### 6.3 Quantization Support

**Data Types** (nsp-model.cpp:996-1013):
- `UFIXED_POINT_8`: 8-bit quantized
- `UFIXED_POINT_16`: 16-bit quantized
- `FLOAT_16`: IEEE 754 half-precision
- `FLOAT_32`: Full precision

---

## 7. Code Examples

### 7.1 Vectorized Causal Attention Mask

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-htp/attention-mask.cpp`

```cpp
template <typename T>
void AttentionMask::fillAttentionRow(
    const std::vector<bool>& attention_row,
    T* attention_buffer,
    uint32_t query_token_idx) const
{
    const T pos_val = 1;
    const uint32_t past_idx = m_kvmanager.m_n_past_idx;
    const uint32_t new_idx = m_kvmanager.m_n_new_idx;
    const uint32_t n_past = m_kvmanager.n_past();
    const uint32_t n_valid_kv = m_kvmanager.n_valid_kv();

    // Vectorized fill for past KV cache
    PRAGMA_LOOP_VECTORIZE
    for (size_t j = 0; j < n_valid_kv; j++) {
        if (attention_row[n_past - n_valid_kv + j]) {
            attention_buffer[past_idx + j] = pos_val;
        }
    }

    // Vectorized fill for new tokens (causal)
    PRAGMA_LOOP_VECTORIZE
    for (size_t j = 0; j <= query_token_idx; j++) {
        if (attention_row[n_past + j]) {
            attention_buffer[new_idx + j] = pos_val;
        }
    }
}
```

### 7.2 HVX L2 Cache Prefetch

**File:** `include/QNN/HTP/core/intrinsics.h`

```cpp
// Prefetch 2D attention matrix into L2 cache
inline void prefetch_attention_matrix(
    const void* q_matrix,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim)
{
    uint32_t height = num_heads;
    uint32_t width = seq_len * head_dim / 128;  // 128-byte vectors
    uint32_t stride = (seq_len * head_dim + 127) / 128;  // Align to vector boundary

    l2pref(q_matrix, height, width, stride);
}
```

### 7.3 64-bit Attention Score Accumulation

**File:** `include/QNN/HTP/core/intrinsics.h`

```cpp
// Accumulate attention scores without overflow
HVX_VectorPair attention_accumulator = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());

for (int i = 0; i < seq_len; i++) {
    HVX_Vector scores = load_attention_scores(i);  // Load 32 float32 scores
    attention_accumulator = addv_u64(attention_accumulator, scores);
}

// Convert to float for softmax
HVX_Vector float_scores = uint64_to_float(attention_accumulator);
```

---

## 8. Intrinsic Operations Summary

### 8.1 Complete Intrinsic Reference Table

| Category | Key Intrinsics | Hardware Op | Attention Use Case |
|----------|----------------|-------------|-------------------|
| **Memory** | `vmemu`, `q6op_vstu_AV`, `q6op_vstu_QAV`, `q6op_vstu_variable_ARV` | Unaligned load/store | Load Q/K/V matrices |
| **Prefetch** | `dcfetch`, `dcfetch_multi`, `l2fetch` | L2 cache prefetch | Pre-load attention data |
| **Arithmetic (Byte)** | `Q6_Vub_vmax_VubVub`, `Q6_Vub_vmin_VubVub` | 8-bit max/min | Quantized score scaling |
| **Arithmetic (Half)** | `Q6_Vh_vmpy_VhRh_s1_rnd_sat`, `Q6_Vh_vasl_VhR` | 16-bit multiply, shift | Fixed-point computation |
| **Arithmetic (Word)** | `Q6_Vw_vadd_VwVw`, `Q6_Vw_vsub_VwVw`, `Q6_Vw_vmpy_VwVw` | 32-bit add/sub/mul | Accumulate scores |
| **Double-Width** | `Q6_W_vcombine_VV`, `Q6_V_lo_W`, `Q6_V_hi_W`, `Q6_W_vdeal_VVR` | 256-bit operations | Process K/V pairs |
| **MAC** | `Q6_Wh_vmpa_WubRb`, `Q6_Wqf32_vmpy_VhfVhf` | 8×8→16, 16×16→32 MAC | Q×K^T matmul |
| **Type Conversion** | `Q6_Vqf32_equals_Vsf`, `uint64_to_float`, `s16_from_hf_core` | Float format conversions | Mixed-precision |
| **Predicates** | `Q6_Q_vcmp_gt_VubVub`, `Q6_Q_vsetq2_R`, `Q6_Vw_condacc_QVwVw` | Conditional ops | Apply causal masks |
| **Saturation** | `Q6_Vh_vround_VwVw_sat`, `Q6_Vh_vlut4_VuhPh` | Saturate, LUT | Clip attention scores |
| **Sync** | `scatter_release_and_stall`, `pause`, `unpause` | Pipeline control | DMA synchronization |

### 8.2 Vector Sizes

- **HVX_Vector**: 128 bytes (1024 bits)
- **HVX_VectorPair**: 256 bytes (2048 bits)
- **Elements per Vector**:
  - `uint8_t`: 128 elements
  - `uint16_t`: 64 elements
  - `uint32_t`: 32 elements
  - `float16`: 64 elements
  - `float32`: 32 elements

---

## 9. GPU Support

### 9.1 GPU Model Implementation

**File:** `examples/Genie/Genie/src/qualla/engines/qnn-gpu/gpu-model.hpp` (165 lines)

**GPU-Specific Features:**

1. **Causal Mask Preparation** (Line 119):
```cpp
void prepareCausalMask(uint16_t* attnMaskBuffer, uint32_t currQuerySize);
```

2. **KV Cache Structure** (Lines 42-54):
```cpp
struct GpuKVCache {
  bool isKey;
  uint32_t tensorId;
  QnnUtils::Tensor* tensorUtil;
};
std::vector<GpuKVCache> _kvCache;
```

**Note:** GPU backend uses device-specific shader languages (GLSL/OpenCL), not HVX intrinsics.

---

## 10. File Reference Index

### 10.1 Critical Files for Custom Kernel Development

| File Path | Lines | Purpose |
|-----------|-------|---------|
| `include/QNN/HTP/core/intrinsics.h` | 826 | Core HVX intrinsics library |
| `include/QNN/HTP/core/hvx_mathops.h` | 85 | HVX math operations |
| `include/QNN/HTP/core/builtin_intrinsics.h` | 248 | Compiler builtin intrinsics |
| `examples/Genie/Genie/src/qualla/engines/qnn-htp/attention-mask.hpp` | 130 | Attention mask interface |
| `examples/Genie/Genie/src/qualla/engines/qnn-htp/attention-mask.cpp` | 233 | Attention mask implementation |
| `examples/Genie/Genie/src/qualla/engines/qnn-htp/KVCache/kvmanager.hpp` | 461 | KV cache orchestration |
| `examples/Genie/Genie/src/qualla/engines/qnn-htp/KVCache/native-kv.hpp` | 94 | HMX-optimized KV layout |
| `examples/Genie/Genie/src/qualla/engines/qnn-htp/KVCache/context-manager.hpp` | 128 | Context management strategies |
| `examples/Genie/Genie/src/qualla/engines/qnn-htp/nsp-model.hpp` | 303 | NSP model interface |
| `examples/Genie/Genie/src/qualla/engines/qnn-htp/nsp-model.cpp` | 2500+ | NSP model implementation |
| `examples/Genie/Genie/src/AUTOREGRESSIVE_TEXT_GENERATION.md` | 1216 | LLM inference documentation |
| `examples/QNN/OpPackage/HTP/ExampleOpPackageRelu.cpp` | 438 | HVX ReLU reference |
| `examples/QNN/OpPackage/HTP/ExampleOpPackageMaxPool.cpp` | 400+ | HVX MaxPool reference |
| `include/QNN/QnnOpDef.h` | 800+ | QNN operation definitions |

### 10.2 Configuration Files

| File Path | Model |
|-----------|-------|
| `examples/Genie/configs/llama2-7b/llama2-7b-genaitransformer.json` | Llama 2 7B |
| `examples/Genie/configs/llama3-8b/llama3-8b-genaitransformer.json` | Llama 3 8B |
| `examples/Genie/configs/phi3-mini/phi3-mini-genaitransformer.json` | Phi 3 Mini |

---

## 11. Performance Characteristics

### 11.1 Attention Mechanism Optimizations

- **Loop Vectorization**: SIMD acceleration via `PRAGMA_LOOP_VECTORIZE`
- **Attention Span Merging**: Reduces iteration overhead
- **Template Specialization**: Data type optimization (uint8/16/32)
- **Causal Mask Pre-computation**: Avoids runtime computation

### 11.2 KV Cache Optimizations

- **Hardware-Aligned Tiles**: K_TILE=256, V_TILE=64 for HMX
- **Smart Masking**: Flexible attention patterns
- **KeyDiff Algorithm**: Intelligent token eviction
- **Sliding Window**: Long-context support

### 11.3 Inference Optimizations

- **Multi-Variant Selection**: AR-1/8/64 for different phases
- **LMHEAD Skipping**: 3× speedup for AR-64 prefill
- **Precomputed RoPE**: O(1) lookup vs O(n) trigonometry
- **Quantized Inference**: 8-bit/16-bit support
- **L2 Prefetching**: Reduces memory latency

---

## 12. Key Takeaways

1. **Production-Ready Implementation**: All components are fully implemented and optimized for Qualcomm hardware.

2. **Hardware Acceleration**:
   - HVX (Hexagon Vector Extensions) for SIMD operations (128-byte vectors)
   - HMX (Hexagon Matrix Extensions) for matrix operations (native KV layout)
   - L2 cache prefetching with stride patterns

3. **Multi-Mode Attention**:
   - CAUSAL: Standard autoregressive attention
   - RELATIONAL: Tree-based sparse patterns
   - CUSTOM: Fully-specified 2D masks

4. **Efficient KV Cache**:
   - Hardware-native layout (K=256, V=64 tiles)
   - Smart masking with flexible patterns
   - Advanced eviction (KeyDiff, SlidingWindow)

5. **Two-Phase Generation**:
   - Prefill: AR-64 for high throughput
   - Decode: AR-1 for low latency
   - Dynamic variant selection

6. **Quantization Support**:
   - 8-bit and 16-bit quantized inference
   - Saturation and rounding intrinsics
   - Mixed-precision operations

7. **Extensive Intrinsics Library**:
   - 826 lines of core HVX intrinsics
   - Unaligned memory operations
   - 64-bit accumulation without overflow
   - Float/int conversion utilities

8. **Example Custom Kernels**:
   - ReLU with HVX vectorization
   - MaxPool with deinterleaving
   - Reference implementations for custom ops

---

## 13. Recommended Development Workflow

### 13.1 For Custom Attention Kernel Development

1. **Study Reference Implementations**:
   - `attention-mask.cpp`: Vectorized mask filling patterns
   - `ExampleOpPackageRelu.cpp`: Basic HVX operations
   - `ExampleOpPackageMaxPool.cpp`: Advanced reduction patterns

2. **Use HVX Intrinsics**:
   - Start with `intrinsics.h` for core operations
   - Use `hvx_mathops.h` for type conversions
   - Apply `l2fetch` for prefetching attention matrices

3. **Optimize Memory Access**:
   - Use `q6op_V_vldu_A` for unaligned loads
   - Use `q6op_vstu_variable_ARV` for boundary conditions
   - Prefetch with stride patterns for 2D matrices

4. **Leverage Vectorization**:
   - Use `PRAGMA_LOOP_VECTORIZE` for compiler auto-vectorization
   - Batch operations in 128-byte chunks
   - Merge adjacent attention spans

5. **Test with Multiple Variants**:
   - AR-1 (single token)
   - AR-8 (small batch)
   - AR-64 (large batch prefill)

### 13.2 For KV Cache Optimization

1. **Choose KV Update Method**:
   - `NATIVE_KV`: For HMX-compatible models (best performance)
   - `SMART_MASK`: For flexible attention patterns

2. **Select Context Manager**:
   - `SlidingWindow`: For bounded long contexts
   - `KeyDiff`: For intelligent token eviction

3. **Configure Tile Sizes**:
   - Align to K_TILE=256, V_TILE=64
   - Ensure multiples of 128 bytes for HVX

---

## 14. Conclusion

The QNN codebase provides a comprehensive, production-ready framework for implementing custom kernels and attention mechanisms for causal language models on Qualcomm hardware. The implementation includes:

- **Extensive HVX intrinsics** for SIMD operations
- **Hardware-optimized KV cache** with native HMX layout
- **Flexible attention mechanisms** (causal, relational, custom)
- **Multi-variant graph system** for efficient inference
- **Advanced context management** for long-context LLMs
- **Quantization support** (8-bit, 16-bit)
- **Reference implementations** for custom operations

All components are optimized for Snapdragon platforms (SM8750 and newer) and support modern LLM architectures including Llama 2/3, Phi 3, and multimodal models.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Qualcomm AI Engine Direct (QNN) Version:** 2.40.0.251030
