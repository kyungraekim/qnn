# Autoregressive Text Generation in Genie Engine

This document provides a comprehensive explanation of how the Genie engine (Qualcomm's QNN-based inference engine) implements autoregressive text generation for causal language models on mobile devices.

## Table of Contents

1. [Overview](#overview)
2. [Graph Architecture](#graph-architecture)
3. [Graph Types and Composition](#graph-types-and-composition)
4. [Graph Loading and Initialization](#graph-loading-and-initialization)
5. [Autoregressive Generation Flow](#autoregressive-generation-flow)
6. [Prefill Phase](#prefill-phase)
7. [Decode Phase](#decode-phase)
8. [KV Cache Management](#kv-cache-management)
9. [Inference Strategy](#inference-strategy)
10. [Performance Optimizations](#performance-optimizations)

---

## Overview

The Genie engine implements autoregressive text generation using a **variant-based multi-graph architecture** where:

- **Multiple graph variants** exist for different token batch sizes (AR-1, AR-8, AR-64, etc.)
- **Graphs can be split** into separate components (embeddings, decoder, LM head)
- **KV cache** enables efficient incremental decoding
- **Dynamic graph selection** optimizes for different phases of generation

### Key Components

```
┌─────────────────┐
│  BasicDialog    │  High-level text generation interface
└────────┬────────┘
         │
┌────────▼────────┐
│   NspEngine     │  Engine abstraction layer
└────────┬────────┘
         │
┌────────▼────────┐
│  QnnNspModel    │  QNN model wrapper with graph management
└────────┬────────┘
         │
┌────────▼────────┐
│  QnnNspGraph    │  Individual graph execution manager
└────────┬────────┘
         │
┌────────▼────────┐
│   KVManager     │  KV cache and inference strategy manager
└─────────────────┘
```

---

## Graph Architecture

### Variant Naming Convention

Graphs follow the naming pattern: **`AR-{variant}_CL-{context_length}`**

- **AR (Attention Ratio/Variant)**: Number of tokens processed per inference step
  - `AR-1`: Single token processing (typical for decode)
  - `AR-8`, `AR-64`: Batch token processing (typical for prefill)
  - `AR-4096` (where `AR == CL`): Full context reprocessing mode

- **CL (Context Length)**: Maximum sequence length the graph supports
  - Common values: `CL-2048`, `CL-4096`, `CL-8192`

### Example Configurations

**Typical LLaMA-style model with split graphs:**
```
Decoder graphs:
  - llama_AR-1_CL-4096_decoder
  - llama_AR-8_CL-4096_decoder
  - llama_AR-64_CL-4096_decoder

LM Head graphs:
  - llama_AR-1_CL-4096_lmhead
  - llama_AR-8_CL-4096_lmhead
  - llama_AR-64_CL-4096_lmhead
```

**Monolithic model (no split):**
```
Combined graphs:
  - llama_AR-1_CL-4096
  - llama_AR-8_CL-4096
  - llama_AR-64_CL-4096
```

---

## Graph Types and Composition

### Graph Type Enumeration

**File**: `qualla/engines/qnn-api/QnnApi.hpp:43`

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

### Automatic Graph Classification

**File**: `qualla/engines/qnn-htp/nsp-graph.cpp:185-239`

The engine automatically detects graph types based on input/output tensor signatures:

```cpp
GraphType GraphVariant::determineGraphType(
    const std::unordered_set<std::string>& cacheGroupPrefixes) {

  bool inputIDExists = getInput("input_ids") != nullptr;
  bool pastKVExists = /* check for past_key_* / past_value_* outputs */;
  bool logitsExists = getOutput("logits") != nullptr;

  // Classification logic:

  // Only input_ids → embedding lookup
  if (inputIDExists && !pastKVExists && !logitsExists)
    return GraphType::LUT;

  // Has KV cache outputs, no logits → decoder layers
  if (!inputIDExists && pastKVExists && !logitsExists) {
    if (matchedAllOutputTensors)
      return GraphType::DECODER_PREFILL;  // Prefill-optimized
    return GraphType::DECODER;            // Standard decoder
  }

  // Only logits output → LM head
  if (!inputIDExists && !pastKVExists && logitsExists)
    return GraphType::LMHEAD;

  // Image features → vision encoder
  if (imageFeaturesExists)
    return GraphType::IMAGE_ENCODER;

  return GraphType::DEFAULT;
}
```

### Graph Split Architecture

The engine supports splitting models into multiple sequential graphs via the `m_nsp_graphs` vector:

**File**: `qualla/engines/qnn-htp/nsp-base-model.hpp:238`

```cpp
std::vector<QnnNspGraph> m_nsp_graphs;
```

Each `QnnNspGraph` represents a **graph split** (or stage) that executes sequentially during inference.

#### Common Split Patterns

**Two-stage split (most common):**
```
m_nsp_graphs[0]: DECODER graphs
  ├── AR-1_CL-4096  (decoder)
  ├── AR-8_CL-4096  (decoder)
  └── AR-64_CL-4096 (decoder)

m_nsp_graphs[1]: LMHEAD graphs
  ├── AR-1_CL-4096  (lmhead)
  ├── AR-8_CL-4096  (lmhead)
  └── AR-64_CL-4096 (lmhead)
```

**Three-stage split (for extreme memory constraints):**
```
m_nsp_graphs[0]: LUT (embedding)
m_nsp_graphs[1]: DECODER (transformer layers)
m_nsp_graphs[2]: LMHEAD (final projection)
```

**Monolithic (no split):**
```
m_nsp_graphs[0]: DEFAULT graphs (everything combined)
  ├── AR-1_CL-4096
  ├── AR-8_CL-4096
  └── AR-64_CL-4096
```

### Graph I/O Tensor Signatures

#### DECODER Graph
**Inputs:**
- `hidden_states`: `[batch, variant, hidden_dim]` - Input embeddings/hidden states
- `position_ids_sin/cos`: `[variant, head_dim]` - RoPE position embeddings
- `attention_mask`: `[variant, context_size]` - Causal attention mask
- `past_key_N` / `past_value_N`: `[batch, num_heads, past_len, head_dim]` - Cached KV for each layer N
- `cache_index`: `[variant]` - Indices for KV cache updates

**Outputs:**
- `hidden_states`: `[batch, variant, hidden_dim]` - Updated hidden states
- `present_key_N` / `present_value_N`: `[batch, num_heads, cache_len, head_dim]` - Updated KV cache

#### LMHEAD Graph
**Inputs:**
- `hidden_states`: `[batch, variant, hidden_dim]` - From decoder
- (Optional) `weight`: `[vocab_size, hidden_dim]` - LM head weights as input

**Outputs:**
- `logits`: `[batch, variant, vocab_size]` - Vocabulary logits

---

## Graph Loading and Initialization

### Model Initialization Flow

**File**: `qualla/engines/qnn-htp.cpp:204-259`

```cpp
bool NspEngine::load() {
  // Create model instance
  _model = std::make_unique<QnnNspModel>(_env, _params);

  // Initialize model (loads graphs from .bin files)
  _model->initializeModel();

  // Validate model I/O
  _model->validateModel();

  // Initialize I/O tensor buffers
  _model->initializeIOTensors();

  // Initialize KV cache managers
  _model->initializeKVManager();

  // Initialize tensor pointers
  _model->initializeTensorPointers();

  // Precompute RoPE embeddings
  _model->calculate_rope_embeddings();

  // Load LM head weights (if configured as input)
  _model->load_lmhead_weight_as_input();

  return true;
}
```

### Graph Organization and Split Detection

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:323-342`

```cpp
// Determine number of splits from loaded graphs
int32_t n_splits = 0;
for (auto& [_, count] : nsp_graph_count) {
  n_splits = std::max(n_splits, count);
}

// Create QnnNspGraph for each split
m_nsp_graphs.reserve(n_splits);
for (int idx = 0; idx < n_splits; idx++) {
  m_nsp_graphs.emplace_back(idx, _env, m_qnnApi.get(), m_ioTensor);
}

// Distribute graph variants across splits
for (auto& [variant_spec, graphs] : graph_names) {
  const auto& [variant, ctx_size] = variant_spec;
  uint32_t idx = 0;

  // Graph names are sorted, so iterate by split index
  for (auto& graph_name : graphs) {
    __INFO("Inserting graph {} as idx {} for AR-{} CL-{}",
           graph_name, idx, variant, ctx_size);
    m_nsp_graphs[idx++].addGraph(m_graph_map.at(graph_name));
  }
}
```

**Example log output:**
```
Inserting graph llama_AR-1_CL-4096_decoder as idx 0 for AR-1 CL-4096
Inserting graph llama_AR-1_CL-4096_lmhead as idx 1 for AR-1 CL-4096
Inserting graph llama_AR-8_CL-4096_decoder as idx 0 for AR-8 CL-4096
Inserting graph llama_AR-8_CL-4096_lmhead as idx 1 for AR-8 CL-4096
```

This shows a **2-split configuration** where:
- Split 0 (idx=0) contains all DECODER variants
- Split 1 (idx=1) contains all LMHEAD variants

### Graph Variant Registration

**File**: `qualla/engines/qnn-htp/KVCache/kvmanager.cpp:177-179`

```cpp
void KVManager::registerSupportedVariant(int32_t variant, int32_t ctx_size) {
  if (ctx_size != -1)
    m_supported_variants[ctx_size].insert(variant);
}
```

The KVManager maintains a map of supported configurations:
```cpp
// Example: m_supported_variants
{
  4096: {1, 8, 64},      // AR-1, AR-8, AR-64 for CL-4096
  8192: {1, 8, 64, 128}  // AR-1, AR-8, AR-64, AR-128 for CL-8192
}
```

---

## Autoregressive Generation Flow

### High-Level Entry Point

**File**: `qualla/dialogs/basic.cpp:133-257`

```cpp
bool BasicDialog::process(std::vector<int32_t>& tokens,
                          DialogCallback callback) {
  // Initialize logits tensor
  Tensor logits;

  // Get engine and sampler references
  auto& engine = *_engine["primary"];
  auto& sampler = *_sampler["primary"];

  // Load model if using dynamic loading
  if (engine.supports(FF::DYNAMIC_LOAD))
    engine.load();

  // ============ PREFILL PHASE ============
  // Process all prompt tokens at once
  const size_t n_returned = engine.process(tokens, logits, false);

  // Update KV cache with all prompt tokens
  _n_past += tokens.size();
  engine.updateKV(_n_past);

  // Sample first generated token
  tokens[0] = sampler.process(logits);
  _n_generated++;

  // ============ DECODE PHASE ============
  // Generate tokens one by one
  processFollowOnGeneration(tokens, logits, callback);

  return true;
}
```

### Complete Generation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUTS PROMPT                     │
│          "What is the capital of France?" → tokens          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      PREFILL PHASE                          │
├─────────────────────────────────────────────────────────────┤
│  1. Validate context: n_past + tokens.size() <= ctx_size    │
│  2. Select inference strategy (e.g., AR-64 for 50 tokens)   │
│  3. Setup input tensors:                                    │
│     - input_ids: [50 tokens]                                │
│     - position_ids: [0..49]                                 │
│     - attention_mask: causal mask                           │
│  4. Execute DECODER graph (AR-64)                           │
│     → Produces hidden_states + KV cache                     │
│  5. Execute LMHEAD graph (AR-64)                            │
│     → Produces logits for last token                        │
│  6. Update KV cache: n_past = 50                            │
│  7. Sample token: "The" (token_id = 450)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   DECODE PHASE (LOOP)                       │
├─────────────────────────────────────────────────────────────┤
│  ITERATION 1:                                               │
│  ├─ Input: token "The" (450)                                │
│  ├─ Select strategy: AR-1                                   │
│  ├─ Setup: position=50, attend to past 50 tokens            │
│  ├─ Execute DECODER (AR-1) → hidden_states                  │
│  ├─ Execute LMHEAD (AR-1) → logits                          │
│  ├─ Update KV: n_past = 51                                  │
│  ├─ Sample: "capital" (token_id = 3297)                     │
│  └─ Callback: emit "The" to user                            │
│                                                             │
│  ITERATION 2:                                               │
│  ├─ Input: token "capital" (3297)                           │
│  ├─ Select strategy: AR-1                                   │
│  ├─ Setup: position=51, attend to past 51 tokens            │
│  ├─ Execute DECODER (AR-1) → hidden_states                  │
│  ├─ Execute LMHEAD (AR-1) → logits                          │
│  ├─ Update KV: n_past = 52                                  │
│  ├─ Sample: "of" (token_id = 310)                           │
│  └─ Callback: emit "capital" to user                        │
│                                                             │
│  ITERATION 3-N: Continue until EOS or max length...         │
│                                                             │
│  FINAL ITERATION:                                           │
│  ├─ Sample: <EOS> (token_id = 2)                            │
│  ├─ Callback: emit final token, then END signal             │
│  └─ Generation complete                                     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT TO USER                           │
│         "The capital of France is Paris."                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Prefill Phase

The prefill phase processes the entire input prompt to initialize the KV cache.

### Prefill Implementation

**File**: `qualla/dialogs/basic.cpp:163-199`

```cpp
// Validate context size
if (_n_past + tokens.size() > _ctx->size()) {
  throw ContextLimitException("Context Size was exceeded.");
}

// Process all prompt tokens
// logits_all=false means only return logits for last token
const size_t n_engine_returned = engine.process(tokens, logits, false);

if (n_engine_returned != 1 || engine.failed()) {
  return Dialog::abort("Engine prompt processing failed.");
}

// Update token checkpoints for all prompt tokens
for (uint32_t idx = 0; idx < tokens.size(); idx++) {
  engine.updateTokenCheckpoint(tokens[idx], _n_past + idx);
}

_n_prompt += tokens.size();
_n_past += tokens.size();

// Update KV cache with all processed tokens
if (!engine.updateKV(_n_past) || engine.failed()) {
  return Dialog::abort("KV cache update failed.");
}

// Sample first generated token from logits
tokens[0] = _last_tok = sampler.process(logits);
sampler.updateSampledTokenHistory(tokens[0]);
tokens.resize(1);
```

### Prefill Characteristics

| Aspect | Details |
|--------|---------|
| **Input size** | N tokens (e.g., 50-200 tokens) |
| **Graph variant** | Large AR (e.g., AR-64) for batch processing |
| **KV cache** | Initialized with all prompt tokens |
| **Attention** | Causal mask over prompt tokens |
| **Output** | Logits for last token only |
| **n_past update** | Incremented by N (prompt length) |
| **Performance** | High throughput (batch processing) |

### Multi-Step Prefill

For very long prompts, the engine may split prefill into multiple steps:

```
Prompt: 200 tokens with AR-64 available

Step 1: Process tokens [0:64]   using AR-64
Step 2: Process tokens [64:128] using AR-64
Step 3: Process tokens [128:192] using AR-64
Step 4: Process tokens [192:200] using AR-8

Total: 4 inference steps
```

---

## Decode Phase

The decode phase generates tokens one at a time in an autoregressive loop.

### Decode Loop Implementation

**File**: `qualla/dialogs/basic.cpp:52-131`

```cpp
bool BasicDialog::processFollowOnGeneration(
    std::vector<int32_t>& tokens,
    Tensor& logits,
    DialogCallback callback) {

  auto& sampler = *_sampler["primary"];
  auto& engine = *_engine["primary"];

  while (true) {
    // Check for cancellation
    if (State::canceled()) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }

    // Validate context size
    if (_n_past + 1 > _ctx->size()) {
      throw ContextLimitException("Context Size was exceeded.");
    }

    // ============ INFERENCE ============
    // Process single token (AR-1 graph selected automatically)
    if (engine.process(tokens, logits, false) != 1 || engine.failed()) {
      return Dialog::abort("Engine processing failed.");
    }

    // ============ SAMPLING ============
    tokens[0] = _last_tok = sampler.process(logits);
    sampler.updateSampledTokenHistory(tokens[0]);

    // ============ STATE UPDATE ============
    _n_past++;
    _n_generated++;
    engine.updateTokenCheckpoint(_last_tok, _n_past);

    // Update KV cache with new token
    if (!engine.updateKV(_n_past)) {
      return Dialog::abort("KV update failed");
    }

    // ============ TERMINATION CHECK ============
    if (_ctx->is_eos(_last_tok)) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }

    // ============ STREAMING OUTPUT ============
    // Emit token to user
    if (!callback.callBack(tokens.data(), tokens.size(),
                           Sentence::CONTINUE, tokenizer())) {
      break;
    }
  }

  return true;
}
```

### Decode Characteristics

| Aspect | Details |
|--------|---------|
| **Input size** | 1 token |
| **Graph variant** | Small AR (AR-1) for single-token processing |
| **KV cache** | Reused from previous tokens, add new entry |
| **Attention** | Attends to all past tokens (0 to n_past) |
| **Output** | Logits for current token |
| **n_past update** | Incremented by 1 |
| **Performance** | Lower throughput (sequential processing) |
| **Streaming** | Tokens emitted immediately via callback |

---

## KV Cache Management

The KV cache is critical for efficient autoregressive generation, avoiding recomputation of attention for past tokens.

### KV Cache Structure

For each transformer layer, the cache stores:
```
past_key_N:   [batch, num_heads, n_past, head_dim]
past_value_N: [batch, num_heads, n_past, head_dim]
```

Where:
- `N` is the layer index
- `n_past` is the number of tokens already processed
- Cache is updated after each inference step

### KV Cache Update Flow

**File**: `qualla/engines/qnn-htp.cpp:272-297`

```cpp
bool NspEngine::updateKV(size_t n_past) {
  if (n_past > _ctx.size()) {
    State::error("context size exceeded");
    throw ContextLimitException();
  }

  // Delegate to model's KV cache manager
  if (!_model->setKVCacheNPast(n_past, selected)) {
    return false;
  }

  return true;
}
```

### Cache Index Management

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:1614-1622`

```cpp
// Set up the input scatter index (where new KV is stored)
for (auto& [prefix, group_index_tensor] : m_group_cache_index) {
  CacheGroup& group = m_kvmanager->getCacheGroups().at(prefix);
  InferenceStep group_step = group.translateInferenceStep(step);

  uint32_t* group_index_buffer =
      reinterpret_cast<uint32_t*>(getBuffer(group_index_tensor));

  // Fill cache_index buffer with [new_idx, new_idx+1, ..., new_idx+n-1]
  std::iota(group_index_buffer,
            group_index_buffer + group_index_tensor->dims.getNumElements(),
            group_step.new_idx);
}
```

### KV Update Methods

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:103-106`

```cpp
enum class KVManagerMode {
  SHIFT_CONCAT  = 0x1,  // Deprecated
  SMART_MASK    = 0x2,  // Intelligent masking (default)
  NATIVE_KV     = 0x3,  // HMX native layout for hardware acceleration
  POINTER_SHIFT = 0x4   // Deprecated
};
```

**SMART_MASK** (default):
- Uses attention masking to manage valid KV entries
- Allows selective KV updates
- Efficient for long context scenarios

**NATIVE_KV**:
- Uses hardware-optimized KV layout (HMX format)
- Detected automatically when graphs use `QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT`
- Provides best performance on supported hardware

---

## Inference Strategy

The inference strategy determines which graph variants to use for processing input tokens.

### Strategy Generation

**File**: `qualla/engines/qnn-htp/KVCache/kvmanager.cpp:360-461`

```cpp
bool KVManager::prepareInferenceStrategy(int32_t n_inputs) {
  std::vector<InferenceStep> strategy;

  // Select context size
  auto iter_ctx = m_supported_variants.lower_bound(n_past + n_inputs);
  if (iter_ctx == m_supported_variants.end()) {
    iter_ctx = m_supported_variants.rbegin().base();
    iter_ctx--;
  }

  int32_t ctx_size = iter_ctx->first;
  int32_t variant = pick(n_inputs, iter_ctx->second);
  int32_t n_remain = n_inputs;

  // Special case: AR == CL means full reprocessing
  if (ctx_size == variant) {
    n_remain += n_past;
    n_past = n_valid_kv = 0;
  }

  // Generate inference steps
  while (n_remain > 0) {
    int32_t n_process = std::min(n_remain, variant);
    int32_t cacheBoundary = ctx_size - variant;

    // Check if we need to switch to larger context
    if (variant != ctx_size && n_valid_kv + variant > cacheBoundary) {
      auto it = m_supported_variants.lower_bound(ctx_size + 1);
      if (it != m_supported_variants.end()) {
        ctx_size = it->first;
        variant = pick(n_remain, it->second);
        n_process = std::min(n_remain, variant);
      }
    }

    const int32_t past_dim = ctx_size - variant;
    strategy.emplace_back(variant, ctx_size, n_past, n_valid_kv,
                          n_process, 0, past_dim);
    strategy.back().new_idx =
        default_group->manager->getIndexForNewKV(strategy.back());

    // Update for next iteration
    n_past += n_process;
    n_valid_kv += n_process;
    n_remain -= n_process;
  }

  m_strategy = strategy;
  m_strategy_cur_step = 0;
  m_strategy_active = true;

  return true;
}
```

### InferenceStep Structure

**File**: `qualla/engines/qnn-htp/KVCache/kvmanager.hpp:36-66`

```cpp
struct InferenceStep {
  int32_t variant{0};      // AR value (graph variant to use)
  int32_t ctx_size{0};     // CL value (context length)
  int32_t n_past{0};       // Number of tokens in KV cache
  int32_t n_valid_kv{0};   // Valid KV entries (may differ with long context)
  int32_t n_process{0};    // Tokens to process in this step
  int32_t past_idx{0};     // Index for past KV
  int32_t new_idx{0};      // Index where new KV will be stored

  std::string str() const {
    return fmt::format("AR-{} CL-{} n_past={} n_valid_kv={} n_process={}",
                       variant, ctx_size, n_past, n_valid_kv, n_process);
  }
};
```

### Strategy Examples

**Example 1: Prefill with 50 tokens, AR-64 available**
```
Strategy:
  Step 1: AR-64 CL-4096 n_past=0 n_valid_kv=0 n_process=50
```

**Example 2: Prefill with 200 tokens, AR-64 available**
```
Strategy:
  Step 1: AR-64 CL-4096 n_past=0   n_valid_kv=0   n_process=64
  Step 2: AR-64 CL-4096 n_past=64  n_valid_kv=64  n_process=64
  Step 3: AR-64 CL-4096 n_past=128 n_valid_kv=128 n_process=64
  Step 4: AR-8  CL-4096 n_past=192 n_valid_kv=192 n_process=8
```

**Example 3: Decode (single token), n_past=50**
```
Strategy:
  Step 1: AR-1 CL-4096 n_past=50 n_valid_kv=50 n_process=1
```

---

## Model Inference Execution

### Main Inference Loop

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:1721-1872`

```cpp
size_t QnnNspModel::runInference(
    const std::vector<int32_t>& tokens,
    std::vector<uint8_t>& embedding,
    ...,
    std::vector<float>& output,
    bool output_all) {

  const size_t n_inputs = tokens.size() + embeddingCount;

  // Construct attention mask processor
  AttentionMask attention_mask(attention_map, n_past, n_valid_kv,
                               n_inputs, ...);

  // ============ GENERATE INFERENCE STRATEGY ============
  if (!m_kvmanager->prepareInferenceStrategy(n_inputs)) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  // Allocate output buffer
  size_t output_size = output_all ? n_inputs : 1;
  output.resize(output_size * m_vocab_size);

  // ============ EXECUTE INFERENCE STEPS ============
  InferenceStep step;
  uint32_t n_processed = 0;

  while (m_kvmanager->nextInferenceStep(step)) {
    __DEBUG("Inference step: {}", step.str());

    // Setup input tensors for this step
    if (!setupInput(step, n_processed, tokens, embedding,
                    featureVector, selected, start_idx,
                    post_update, attention_mask)) {
      return false;
    }

    // ============ EXECUTE ALL GRAPH SPLITS ============
    for (auto& nsp_graph : m_nsp_graphs) {
      const int graph_idx = nsp_graph.idx();

      // Optimization: Skip LMHEAD for intermediate steps
      if (nsp_graph.m_graphType == GraphType::LMHEAD &&
          (!output_all) &&
          !m_kvmanager->isFinalInferenceStep()) {
        continue;  // Skip LMHEAD execution
      }

      // Block until graph is ready
      if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }

      // Execute the graph
      if (!nsp_graph.execute(step.variant, step.ctx_size,
                             m_inference_count,
                             graph_switching, lazy_lora)) {
        fatal(fmt::format("Failed to execute graph {}", graph_idx));
        return false;
      }

      // Unblock for next graph
      if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }
    }

    m_kvmanager->completeInferenceStep();

    // Extract logits if needed (intermediate steps with output_all)
    if (output_all) {
      getDequantLogits(
          std::span(&output[n_processed * m_vocab_size],
                    step.n_process * m_vocab_size),
          step, step.n_process);
    }

    n_processed += step.n_process;
    m_inference_count++;
  }

  // Extract final logits
  if (!output_all) {
    getDequantLogits(std::span{output.data(), output.size()},
                     step, 1);
  }

  return output_size;
}
```

### Input Setup

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:1520-1694`

```cpp
bool QnnNspModel::setupInput(InferenceStep& step,
                             uint32_t start,
                             const std::vector<int32_t>& tokens,
                             ...) {
  const int32_t variant = step.variant;
  const int32_t n_past = step.n_past;
  const int32_t n_process = step.n_process;

  // ============ SETUP TOKEN INPUT ============
  if (!tokens.empty()) {
    for (uint32_t i = 0; i < n_process; i++) {
      // Copy token to input_ids buffer
      std::copy(tokenPtr, tokenPtr + id_bytes, outputPtr);
    }
  }

  // ============ SETUP CACHE INDICES ============
  for (auto& [prefix, group_index_tensor] : m_group_cache_index) {
    uint32_t* group_index_buffer =
        reinterpret_cast<uint32_t*>(getBuffer(group_index_tensor));

    // Fill with [new_idx, new_idx+1, ..., new_idx+n_process-1]
    std::iota(group_index_buffer,
              group_index_buffer + group_index_tensor->dims.getNumElements(),
              group_step.new_idx);
  }

  // ============ SETUP ATTENTION MASK ============
  setupAttentionMask<uint16_t>(step, attention_mask);

  // ============ SETUP POSITION EMBEDDINGS ============
  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    // Get position IDs for current step
    const auto& position_ids =
        attention_mask.getPositionIds(start, n_process, variant);

    // Copy precomputed sin/cos embeddings
    uint8_t* cos_buffer = reinterpret_cast<uint8_t*>(
        getBuffer(t_position_ids_cos));
    uint8_t* sin_buffer = reinterpret_cast<uint8_t*>(
        getBuffer(t_position_ids_sin));

    const size_t rope_size = m_pos_dim * d_pos.bw();
    for (uint32_t i = 0; i < variant; i++) {
      const size_t src_offset = position_ids[i] * rope_size;
      const size_t dst_offset = i * rope_size;
      std::memcpy(&sin_buffer[dst_offset],
                  reinterpret_cast<uint8_t*>(rope_sin) + src_offset,
                  rope_size);
      std::memcpy(&cos_buffer[dst_offset],
                  reinterpret_cast<uint8_t*>(rope_cos) + src_offset,
                  rope_size);
    }
  }

  return true;
}
```

### Graph Execution

**File**: `qualla/engines/qnn-htp/nsp-graph.cpp:297-400`

```cpp
bool QnnNspGraph::execute(int32_t variant, int32_t ctx_size,
                          int n_inference, ...) {
  // Get the appropriate graph variant
  GraphVariant* graph = (*this)(variant, ctx_size);
  if (!graph) {
    __ERROR("Graph AR-{} CL-{} not found", variant, ctx_size);
    return false;
  }

  // Execute QNN graph
  Qnn_ErrorHandle_t error = g_qnn_api->qnnInterface.graphExecute(
      graph->graph_info->graph,
      graph->graph_info->inputTensors,
      graph->graph_info->numInputTensors,
      graph->graph_info->outputTensors,
      graph->graph_info->numOutputTensors,
      nullptr,  // profileHandle
      nullptr); // signalHandle

  if (error != QNN_SUCCESS) {
    __ERROR("QNN graph execution failed: {}", error);
    return false;
  }

  return true;
}
```

### Logit Extraction

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:2127-2177`

```cpp
size_t QnnNspModel::getDequantLogits(std::span<float> buffer,
                                     InferenceStep& step,
                                     int32_t count) {
  // Get output tensor from LAST graph split (LMHEAD)
  QnnUtils::Tensor* const spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)
          ->getOutput(m_layerNames[LayerType::OUTPUT]);

  if (spec == nullptr) {
    State::error("Failed to get output layer tensor spec");
    return 0;
  }

  auto [scale, offset] = spec->quantParam[0];
  QnnUtils::DataType dtype(spec->tensor);
  uint32_t bitwidth = spec->dtype.bw();

  auto logit_buffer = reinterpret_cast<uint8_t*>(getBuffer(spec));

  // Offset to last token's logits (right-padded input)
  logit_buffer += (step.n_process - count) * bitwidth * m_vocab_size;

  const size_t size = m_vocab_size * count;

  // Dequantize/convert to FP32
  switch (dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      deQuantizeOutputs(reinterpret_cast<uint8_t*>(logit_buffer),
                        buffer, scale, offset, size);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      deQuantizeOutputs(reinterpret_cast<uint16_t*>(logit_buffer),
                        buffer, scale, offset, size);
      break;
    case QNN_DATATYPE_FLOAT_16:
      castOutputs(reinterpret_cast<uint16_t*>(logit_buffer),
                  buffer, size, bitwidth);
      break;
    case QNN_DATATYPE_FLOAT_32:
      castOutputs(reinterpret_cast<float*>(logit_buffer),
                  buffer, size, bitwidth);
      break;
  }

  return size;
}
```

---

## Performance Optimizations

### 1. LMHEAD Graph Skipping

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:1809-1812`

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

**Benefits:**
- For multi-step prefill (e.g., 256 tokens as 4×AR-64):
  - Step 1-3: Skip LMHEAD (3× savings)
  - Step 4: Execute LMHEAD only once
- Reduces computation by ~75% for long prompts

### 2. Dynamic Graph Selection

The engine automatically selects the optimal graph variant based on:
- Number of tokens to process
- Available graph variants
- Current KV cache state

**Example:**
```
50 tokens to process, available: AR-1, AR-8, AR-64
→ Select AR-64 (process all 50 in one step)

5 tokens to process, available: AR-1, AR-8, AR-64
→ Select AR-8 (better than AR-64 for small batch)

1 token to process (decode)
→ Select AR-1 (optimal for single token)
```

### 3. KV Cache Reuse

During decode phase:
- Only new token's KV computed
- Previous tokens' KV reused from cache
- Avoids O(n²) recomputation
- Reduces compute per token to O(1) amortized

### 4. Quantized Inference

Supports multiple quantization formats:
- `UFIXED_POINT_8`: 8-bit quantized
- `UFIXED_POINT_16`: 16-bit quantized
- `FLOAT_16`: FP16
- `FLOAT_32`: FP32

Quantization reduces:
- Memory bandwidth
- Storage requirements
- Computation time

### 5. Hardware-Optimized KV Layout

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:348-357`

```cpp
// Detect HMX weight layout format
if (tspec.tensor->v1.dataFormat == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT) {
  _kv_update_method = KVManagerMode::NATIVE_KV;
  m_expectedDataFormat = tspec.tensor->v1.dataFormat;
  m_qnnApi->setKVUpdateMethod(_kv_update_method);
}
```

**NATIVE_KV mode:**
- Uses hardware-native KV cache format
- Optimized for Qualcomm HMX (Hexagon Matrix Extensions)
- Eliminates format conversion overhead

### 6. Precomputed RoPE Embeddings

**File**: `qualla/engines/qnn-htp/nsp-model.cpp:2291-2413`

```cpp
bool QnnNspModel::calculate_rope_embeddings(void) {
  const size_t nmemb = m_ctx_size * m_pos_dim;

  rope_sin = malloc(nmemb * pos_bw);
  rope_cos = malloc(nmemb * pos_bw);

  // Precompute for all positions [0, ctx_size)
  for (uint32_t i = 0; i < m_ctx_size; i++) {
    for (uint32_t j = 0; j < m_pos_dim; j++) {
      const double freq = i * inv_freq[j];
      rope_sin[i * m_pos_dim + j] = sin(freq);
      rope_cos[i * m_pos_dim + j] = cos(freq);
    }
  }

  return true;
}
```

**Benefits:**
- Computed once during initialization
- Lookup during inference (no trigonometry)
- Supports RoPE scaling (LLaMA3, LongRope)

### 7. Streaming Token Generation

**File**: `qualla/dialogs/basic.cpp:119`

```cpp
// Emit token immediately after generation
if (!callback.callBack(tokens.data(), tokens.size(),
                       Sentence::CONTINUE, tokenizer())) {
  break;
}
```

**Benefits:**
- Low latency (user sees tokens as generated)
- No buffering delay
- Allows early termination

---

## Performance Metrics

The engine tracks detailed KPIs for both phases:

**File**: `qualla/include/qualla/dialog.hpp:92-118`

```cpp
struct KPIs {
  struct Kpi {
    size_t samples{0};
    float mean{0.f};
    float last{0.f};
  };

  Kpi prompt;      // Prefill phase metrics
  Kpi generate;    // Decode phase metrics

  struct Tps {
    size_t n_prompt;
    size_t n_generate;
    float prompt;          // Tokens/sec during prefill
    float generate;        // Tokens/sec during decode
    float tokenAcceptance;
  } tps{0};

  std::string dump(const std::string& sep) const {
    return fmt::format(
        "prompt:{:.2f}ms{}"
        "generate:{:.2f}ms{}"
        "tps:prompt={:.2f}{}"
        "tps:generate={:.2f}",
        prompt.last / 1000.0f, sep,
        generate.last / 1000.0f, sep,
        tps.prompt, sep,
        tps.generate);
  }
};
```

**Example output:**
```
prompt:250.50ms generate:1850.25ms tps:prompt=199.60 tps:generate=24.32
```

---

## Summary

The Genie engine implements autoregressive text generation through:

1. **Multi-graph architecture** with variant-based selection (AR-1, AR-8, AR-64, etc.)
2. **Graph splitting** capability (LUT, DECODER, LMHEAD) for memory efficiency
3. **Two-phase generation**:
   - **Prefill**: Batch-process prompt tokens with large AR variants
   - **Decode**: Sequential single-token generation with AR-1
4. **KV cache management** for efficient incremental decoding
5. **Dynamic inference strategy** that adapts to input size and available resources
6. **Hardware optimizations** including quantization, native KV format, and HMX acceleration

This architecture enables efficient LLM inference on mobile devices by:
- Minimizing memory usage through graph splitting
- Maximizing throughput through variant selection
- Reducing redundant computation via KV caching
- Leveraging hardware acceleration features

The result is a flexible, efficient, and production-ready system for running large language models on resource-constrained mobile platforms.
