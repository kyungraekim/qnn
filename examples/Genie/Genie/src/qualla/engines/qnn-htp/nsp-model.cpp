//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#define _USE_MATH_DEFINES  // Used for M_PI

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <sstream>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif  // defined(__GNUC__) || defined(__clang__)
#include "fp16/fp16.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) || defined(__clang__)

#include "Trace.hpp"
#include "TraceLogger.hpp"
#include "attention-mask.hpp"
#include "fmt/format.h"
#include "fmt/os.h"
#include "fmt/ranges.h"
#include "native-kv.hpp"
#include "nsp-model.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/env.hpp"
#include "smart-mask.hpp"

#define __ERROR(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace fs = std::filesystem;

namespace qualla {

QnnNspModel::QnnNspModel(std::shared_ptr<Env> env, const QnnNspBaseModel::Params& params)
    : QnnNspBaseModel(env, params) {
  GENIE_TRACE();
  spill_fill_buffer_size  = params.spill_fill_bufsize;
  m_kv_dim                = params.kv_dim;
  m_use_mmap              = params.use_mmap;
  mmap_budget             = params.mmap_budget;
  m_dataAlignmentSize     = params.data_alignment_size;
  m_ctx_size              = params.ctx_size;
  m_pad_token             = params.pad_token;
  lmhead_weight_dir       = params.lmhead_weight_dir;
  graph_switching         = params.graph_switching;
  lazy_lora               = params.lazy_lora;
  skip_lora_validation    = params.skip_lora_validation;
  load_select_graphs      = params.load_select_graphs;
  embedding_length        = params.embedding_length;
  embedding_datatype      = params.embedding_datatype;
  m_disableKvCache        = params.disable_kv_cache;
  m_embd_size             = params.n_embd;
  m_modelArchitectureType = params.modelArchitectureType;
  m_positional_encoding   = params.positional_encoding_params;

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    m_pos_dim = static_cast<uint32_t>(m_positional_encoding.rope_params.dims);
  }

  // Longcontext params
  m_default_group          = params.default_group;
  m_cache_group_params_map = params.cache_group_params;

  m_draft_tok_map = params.draft_tok_map;
  if (graph_switching && !m_use_mmap)
    __WARN("Graph switching with non-mmaped implementation can cause high sustained memory usage");

  variant_latency = params.variant_latency;

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    m_pooled_output = params.pooled_output;
  }

  exec_select_graphs = params.exec_select_graphs;
  if (!exec_select_graphs.empty())
    __DEBUG("qnn-htp : Execute selected graphs = {}", exec_select_graphs);

  if (params.kv_update_method == "SHIFT_CONCAT" || (params.kv_update_method == "POINTER_SHIFT"))
    __WARN("kv-update-method is deprecated. Defaulting to SMART_MASK or NATIVE_KV");
  _kv_update_method =
      KVManagerMode::SMART_MASK;  // Updates to NATIVE_KV if HMX_WEIGHT_LAYOUT tensor is found

  // Set up filename list.
  for (auto& i : params.model_list) {
    fs::path model_path = fs::path(i);
    if (model_path.is_relative()) model_path = model_basedir / fs::path(i);
    if (!fs::is_regular_file(model_path)) {
      __ERROR("NSPModel: Can't access model file : {}", model_path.string());
      throw std::runtime_error("NSPModel: Can't access model file : " + model_path.string());
    }
    model_filelist.push_back(model_path.string());
  }

  m_qnnApi->setKVDim(static_cast<uint32_t>(m_kv_dim));
  m_qnnApi->setContextSize(m_ctx_size);
  m_qnnApi->setKVUpdateMethod(_kv_update_method);
  m_qnnApi->setDataAlignmentSize(m_dataAlignmentSize);

  for (const auto& [prefix, _] : m_cache_group_params_map) {
    m_cache_group_prefixes.insert(prefix);
  }
  m_qnnApi->setCacheGroupPrefixes(m_cache_group_prefixes);

  if (params.debug_specs || params.debug_tensors) {
    if (!fs::exists(params.debug_path) && !fs::create_directories(params.debug_path))
      throw std::runtime_error("Could not create debug directory : " + params.debug_path);
  }

  // Instantiation of threapool must be done at last, To avoid owner less state of it.
  if (params.n_threads > 0) {
    __DEBUG("nsp-model: starting threadpool : n_threads {} params. {:#x} poll {}",
            params.n_threads,
            params.cpumask,
            params.poll);
    m_threadpool = std::make_shared<ThreadPool>();
    m_threadpool->start(params.n_threads, params.cpumask, params.poll);
  }
}

QnnNspModel::~QnnNspModel() {
  qualla::Timer start;

  // The threadpool needs to be stopped before KVManager
  // destruction to avoid race conditions.
  if (m_kvmanager) {
    m_kvmanager->deRegisterAll();
  }

  // Free cached RoPE memory
  if (rope_sin != nullptr) free(rope_sin);
  if (rope_cos != nullptr) free(rope_cos);

  if (eagle_extra_feature != nullptr) {
    free(eagle_extra_feature);
    eagle_extra_feature = nullptr;
  }
  _counter = nullptr;
  if (m_threadpool) m_threadpool->stop();
  __DEBUG("qnn-htp: model destruct complete: {} usec", start.elapsed_usec());
}

// Given a filename, initializeModel load and initializes QNN runtime libraries and the model
bool QnnNspModel::initializeModel(void) {
  GENIE_TRACE();
  qualla::Timer start;

  __DEBUG("qnn-htp: model init start");

  // Default backends
#ifdef _WIN32
  const std::string m_backend                = _backend_lib.empty() ? "QnnHtp.dll" : _backend_lib;
  const std::string m_systemLib              = "QnnSystem.dll";
  const std::string backendExtensionsLibPath = "QnnHtpNetRunExtensions.dll";
#else
  const std::string m_backend                = _backend_lib.empty() ? "libQnnHtp.so" : _backend_lib;
  const std::string m_systemLib              = "libQnnSystem.so";
  const std::string backendExtensionsLibPath = "libQnnHtpNetRunExtensions.so";
#endif

  if (_backend_ext_conf.empty()) {
    __INFO("No backend extension config provided");
  }
  fs::path m_backendExtensionsConfigPath = fs::path(_backend_ext_conf);

  __INFO("Backend library : {}", m_backend);
  __INFO("System library  : {}", m_systemLib);
  __INFO("Model dir   : {}", model_basedir.string());
  __INFO("Model files : {}", model_filelist);
  __INFO("Backend extensions lib path : {}", backendExtensionsLibPath);
  __INFO("Backend extensions config path : {}", m_backendExtensionsConfigPath.string());

  auto logger       = _env->logger();
  uint32_t logLevel = 1;  // error
  std::function<void(const char* fmt, uint32_t level, uint64_t timestamp, va_list args)>
      logCallback = nullptr;
  if (_debug_qnn && logger) {
    logLevel                          = static_cast<uint32_t>(logger->getMaxLevel());
    GenieLog_Callback_t localCallback = logger->getCallback();
    GenieLog_Handle_t localHandle     = logger->getHandle();
    logCallback                       = [localCallback, localHandle](
                      const char* fmt, uint32_t level, uint64_t timestamp, va_list args) {
      // Convert the parameters to match the GenieLog_Callback_t signature
      GenieLog_Level_t genieLevel = static_cast<GenieLog_Level_t>(level);
      localCallback(localHandle, fmt, genieLevel, timestamp, args);
    };
  }
  if (!m_qnnApi->populateGraphBinaryInfo(model_filelist, graph_switching, m_systemLib)) {
    __ERROR("populateGraphBinaryInfo failed");
    return false;
  }

  if (_debug_specs) dumpTensorSpecs();

  // Compile the number of LLM graphs and auxiliary graphs
  const size_t num_graphs = static_cast<size_t>(m_qnnApi->getGraphsCount());
  const auto graphs_info  = m_qnnApi->getGraphsInfo();

  __INFO("qnn-api initialized with {} graph(s)", num_graphs);

  // Finalize the CacheGroup config, filling in missing values with detected tensors
  // We run one pass across all input tensors for all graphs
  for (size_t graph_idx = 0; graph_idx < num_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];
    for (size_t tensor_idx = 0; tensor_idx < graph_info->numInputTensors; tensor_idx++) {
      std::string tname = QNN_TENSOR_GET_NAME(graph_info->inputTensors[tensor_idx]);
      for (auto& [prefix, param] : m_cache_group_params_map) {
        // For any empty tensor name in the cache group configuration, match this schema:
        //    - For the default-group, match either prefix.*m_layerNames or just m_layerNames
        //    - For all other groups, must match prefix.*m_layerNames
        if (param.attention_mask_tensor_name.empty()) {
          if ((prefix == m_default_group && tname == m_layerNames[LayerType::ATTN_MASK]) ||
              (tname.starts_with(prefix) &&
               tname.find(m_layerNames[LayerType::ATTN_MASK]) != std::string::npos)) {
            param.attention_mask_tensor_name = tname;
          }
        }
        if (param.cache_index_tensor_name.empty()) {
          if ((prefix == m_default_group && tname == m_layerNames[LayerType::CACHE_INDEX]) ||
              (tname.starts_with(prefix) &&
               tname.find(m_layerNames[LayerType::CACHE_INDEX]) != std::string::npos)) {
            param.cache_index_tensor_name = tname;
          }
        }
      }
    }
  }

  {
    qualla::json j = m_cache_group_params_map;
    __DEBUG("Detected CacheGroup parameters = {}", j.dump());
  }

  for (auto& [prefix, param] : m_cache_group_params_map) {
    if (param.attention_mask_tensor_name.empty()) {
      __WARN("Could not find attention mask tensor for CacheGroup {}", prefix);
      if (prefix == m_default_group) {
        State::error(fmt::format("Default Group {} has no associated attention mask", prefix));
        return false;
      }
    }
    if (param.cache_index_tensor_name.empty()) {
      __DEBUG("Could not find cache index tensor for CacheGroup {}", prefix);
    }
  }

  m_variant_list.reserve(num_graphs);
  std::map<std::pair<int32_t, int32_t>, std::set<std::string>> graph_names;
  for (size_t graph_idx = 0; graph_idx < num_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];
    const std::string graph_name                   = std::string(graph_info->graphName);

    __DEBUG("qnn-htp: Graph {}", graph_name);
    GraphVariant graph(graph_info, m_layerNames, _env, m_cache_group_prefixes, m_default_group);
    if (!variant_latency.empty() && !variant_latency.contains(graph.n_tokens)) {
      __WARN("qnn-htp: Disabling {} based on conf file", graph_name);
      continue;
    }
    if (exec_select_graphs.size() != 0 &&
        std::find(exec_select_graphs.begin(), exec_select_graphs.end(), graph_name) ==
            exec_select_graphs.end()) {
      __DEBUG("qnn-htp: Graph {} is not selected to execute based on conf file", graph_name);
      continue;
    }
    m_variant_list.emplace_back(graph);
    m_graph_map[graph_name] = &m_variant_list.back();

    std::pair<int32_t, int32_t> variant_spec = {graph.n_tokens, graph.ctx_size};
    nsp_graph_count[variant_spec]++;
    graph_names[variant_spec].insert(graph_name);
  }
  // Collect all available ctx_sizes so we can handle not being able to detect ctx_size in a variant
  std::unordered_set<int32_t> available_ctx_size;
  for (const auto& [variant_spec, count] : nsp_graph_count) {
    if (variant_spec.second != -1) available_ctx_size.insert(variant_spec.second);
  }
  std::vector<std::pair<int32_t, int32_t>> keysToDelete;
  // For all variants where we did not detect a ctx_size, add it to all ctx_sizes
  for (const auto& [variant_spec, count] : nsp_graph_count) {
    if (variant_spec.second != -1) continue;
    auto& prev_names = graph_names.at(variant_spec);
    for (const auto& new_ctx : available_ctx_size) {
      std::pair<int32_t, int32_t> new_spec = {variant_spec.first, new_ctx};
      nsp_graph_count[new_spec]++;
      auto& new_names = graph_names[new_spec];
      new_names.insert(prev_names.begin(), prev_names.end());
    }
    keysToDelete.push_back(variant_spec);
  }
  for (const auto& key : keysToDelete) {
    graph_names.erase(key);
    nsp_graph_count.erase(key);
  }

  if (exec_select_graphs.size() != 0 && graph_names.empty()) {
    __ERROR("No matching graphs based on conf file");
  }

  // Create NSPGraph for each splits
  int32_t n_splits = 0;
  for (auto& [_, count] : nsp_graph_count) {
    n_splits = std::max(n_splits, count);
  }
  m_nsp_graphs.reserve(static_cast<uint32_t>(n_splits));
  for (int idx = 0; idx < n_splits; idx++) {
    m_nsp_graphs.emplace_back(idx, _env, m_qnnApi.get(), m_ioTensor);
    m_nsp_graphs.back().setDebugMode(_debug_specs, _debug_tensors, _debug_path);
  }

  // Insert all GraphVariants into corresponding NSPGraph
  for (auto& [variant_spec, graphs] : graph_names) {
    const auto& [variant, ctx_size] = variant_spec;
    uint32_t idx = 0;  // Graph names are sorted by default (std::set<>), so iterate by split
    for (auto& graph_name : graphs) {
      __INFO("Inserting graph {} as idx {} for AR-{} CL-{}", graph_name, idx, variant, ctx_size);
      m_nsp_graphs[idx++].addGraph(m_graph_map.at(graph_name));
    }
  }

  // Detect whether NATIVE_KV needs to be activated
  for (auto& variant : m_variant_list) {
    m_graph_variant_type_map[variant.graph_name] = variant.variantType;
    for (auto& [tname, tspec] : variant.input_specs) {
      // If QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT is detected, we switch to NATIVE_KV format
      if (tspec.tensor->v1.dataFormat == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT) {
        _kv_update_method    = KVManagerMode::NATIVE_KV;
        m_expectedDataFormat = tspec.tensor->v1.dataFormat;
        m_qnnApi->setKVUpdateMethod(_kv_update_method);
        break;
      }
    }
    if (_kv_update_method == KVManagerMode::NATIVE_KV) break;
  }

  __INFO("qnn-htp: Graphs loaded ((AR-n, CL-x): #splits): {}", nsp_graph_count);

  size_t max_ctx_size = 0;
  for (auto& [variant_spec, count] : nsp_graph_count) {
    max_ctx_size = std::max(max_ctx_size, static_cast<size_t>(variant_spec.second));
  }

  // If LongContext is disabled, make sure the config CL matches loaded CL
  if (max_ctx_size < m_ctx_size && !isLongContextEnabled()) {
    State::error(fmt::format(
        "Config specifies context->size={}, but loaded max-CL={}", m_ctx_size, max_ctx_size));
    return false;
  }

  if (!analyzeCacheGroupKV()) {
    return false;
  }
  m_qnnApi->setGraphVariantType(m_graph_variant_type_map);
  m_qnnApi->setCacheGroupCtxSize(m_cache_group_ctx_size);
  if (!m_qnnApi->initializeHtp(m_backend,
                               model_filelist,
                               BackendExtensionsConfigs(backendExtensionsLibPath,
                                                        m_backendExtensionsConfigPath.string()),
                               {},           // graphConfigs
                               true,         // loadFromCachedBinary
                               m_systemLib,  // systemLibraryPath
                               false,
                               static_cast<size_t>(spill_fill_buffer_size),
                               m_use_mmap,
                               m_use_async_Init,
                               mmap_budget,
                               _debug_qnn,
                               graph_switching,
                               exec_select_graphs,
                               load_select_graphs,
                               skip_lora_validation,
                               m_lazyInitialization,
                               logLevel,
                               logCallback)) {
    __ERROR("qnn-api initialization failed!");
    return false;
  }
  __DEBUG("qnn-htp: Model Init complete: {} usec", start.elapsed_usec());
  return true;
}

// Once the model has been loaded, initialize IO Tensors
// m_ioTensors is initialized by the context for now
bool QnnNspModel::initializeIOTensors() {
  GENIE_TRACE();
  // IO Tensor Mem Registration is already done within the
  // model_initailize by Qnn_API for Sync Init.
  if (m_lazyInitialization) return true;
  // set lmHeadWeightsEnabled and loraWeights Enabled
  _lmhead_weight_input = m_qnnApi->getLmHeadWeightInputEnabled();
  _lora_enabled        = m_qnnApi->getLoraWeightEnabled();
  for (auto it = nsp_graph_count.rbegin(); it != nsp_graph_count.rend(); ++it) {
    for (QnnNspGraph& graph : m_nsp_graphs) {
      // TensorAllocInfo is added to each NSP graph.
      // Needed by Pointer_SHIFT Registration During Execute.
      graph.tensor_alloc_info = m_qnnApi->getTensorAllocInfo();
      graph.g_buffer_mgr      = m_ioTensor;
      if (graph.tensor_alloc_info == NULL) {
        __ERROR("Error Tensor Allocation Failed.");
        return false;
      }
    }
  }

  return true;
}

/* Converts "don't care" dimensions into "*" */
static std::string translateDim(int32_t dim) { return (dim == -1) ? "*" : std::to_string(dim); }

static bool checkShape(const std::string& tensor_name,
                       const QnnUtils::Tensor* tensor,
                       int32_t height,
                       int32_t width,
                       int32_t channel,
                       int32_t bitwidth,
                       std::vector<std::tuple<std::string, std::string, std::string>>& errors) {
  if (tensor != nullptr) {
    const QnnUtils::Dims& tDims = tensor->dims;
    if ((height == -1 || static_cast<uint32_t>(height) == tDims.height) &&
        (width == -1 || static_cast<uint32_t>(width) == tDims.width) &&
        (channel == -1 || static_cast<uint32_t>(channel) == tDims.channel) &&
        (bitwidth == -1 || static_cast<uint32_t>(bitwidth) == tDims.bitwidth)) {
      return true;
    }

    std::stringstream err_msg;
    err_msg << "Expected [ " << translateDim(height) << ", " << translateDim(width) << ", "
            << translateDim(channel) << "] "
            << "bitwidth=" << translateDim(bitwidth) << ". Found [ " << tDims.height << ", "
            << tDims.width << ", " << tDims.channel << "] "
            << "bitwidth=" << tDims.bitwidth;

    errors.push_back({"ShapeError", tensor_name, err_msg.str()});
  }

  return false;
}

// Utility functions
auto toInput  = [](const std::string& s) { return QnnUtils::replaceSubstring(s, "_out", "_in"); };
auto toOutput = [](const std::string& s) { return QnnUtils::replaceSubstring(s, "_in", "_out"); };
auto toVal = [](const std::string& s) { return QnnUtils::replaceSubstring(s, "_key", "_value"); };

bool QnnNspModel::analyzeCacheGroupKV() {
  GENIE_TRACE();
  // Detect if the model uses Scatter (new_key -> past_key) or Concat (past_key + new_key)
  // We can iterate through all graph variants (arn/ctx_size) until we find one with KV$ input
  // If no KV$ input is found, it's AR-c only model and doesn't use Scatter or Concat
  for (const auto& [prefix, param] : m_cache_group_params_map) {
    m_cache_group_use_scatter[prefix] = false;

    // Initialize CacheGroup variant map to a default global->global mapping
    m_cache_group_variant_map[prefix] = {};
    for (auto& nsp_graph : m_nsp_graphs) {
      for (auto& [global_variant, variant] : nsp_graph.variants) {
        m_cache_group_variant_map[prefix][global_variant] = global_variant;
      }
    }

    if (param.attention_mask_tensor_name.empty()) continue;

    // Detect whether KV$ uses Scatter or Concat
    bool detected = false;
    for (const auto& [variant_spec, count] : nsp_graph_count) {
      const auto& [n_tokens, ctx_size] = variant_spec;

      int32_t kv_ctx{0};
      QnnUtils::Tensor* attention_mask = nullptr;
      for (auto& graph : m_nsp_graphs) {
        if (!graph.variants.contains({n_tokens, ctx_size})) continue;
        GraphVariant* variant = graph(n_tokens, ctx_size);

        if (kv_ctx == 0) {
          for (auto& [tname, tspec] : variant->input_specs) {
            if (tname.starts_with(prefix) && tname.find("key") != std::string::npos) {
              kv_ctx = static_cast<int32_t>(tspec.dims.channel);
              break;
            }
          }
        }

        if (attention_mask == nullptr) {
          attention_mask = variant->getInput(param.attention_mask_tensor_name);
        }

        if (kv_ctx != 0 && attention_mask != nullptr) {
          int32_t group_ctx = static_cast<int32_t>(attention_mask->dims.getMaxDim());
          if (kv_ctx == group_ctx) {
            m_cache_group_use_scatter[prefix] = true;
          } else if (kv_ctx == group_ctx - n_tokens) {
            m_cache_group_use_scatter[prefix] = false;
          } else {
            std::string err_msg = "Could not determine whether KV$ uses Scatter or Concat. ";
            err_msg += fmt::format("KV$ has input dimension {}.", kv_ctx);
            err_msg += fmt::format("Expected CL={} or CL - AR-n={}", ctx_size, ctx_size - n_tokens);
            QNN_ERROR("%s", err_msg.c_str());
            State::error(err_msg);
            return false;
          }
          m_cache_group_ctx_size[prefix] = size_t(group_ctx);
          detected                       = true;
          break;
        }
      }
      if (detected) break;
    }

    // Iterate across all [AR-n, CL] to determine variant mapping
    __DEBUG("Mapping for Cachegroup {}", prefix);
    bool found = false;
    for (auto& nsp_graph : m_nsp_graphs) {
      auto& graph_outputs = nsp_graph.variants.begin()->second->output_specs;
      for (auto& [tname, tensor] : graph_outputs) {
        if (!tname.starts_with(prefix) || tname.find("key") == std::string::npos) {
          continue;
        }

        // Found representative tensor for this cache group
        found = true;

        const std::string keyout_name = tname;
        const std::string keyin_name  = toInput(tname);

        for (auto& [global_variant, variant] : nsp_graph.variants) {
          const auto& [global_arn, global_ctx] = global_variant;

          QnnUtils::Tensor* key_out = variant->getOutput(keyout_name);
          QnnUtils::Tensor* key_in  = variant->getInput(keyin_name);

          // Key Cache has shape input[n_heads, n_embed, n_ctx] + output [n_heads, n_embed, arn]
          bool scatter = m_cache_group_use_scatter.at(prefix);
          int32_t arn  = static_cast<int32_t>(key_out->dims.channel);
          int32_t ctx =
              !key_in ? arn : (static_cast<int32_t>(key_in->dims.channel) + (scatter ? 0 : arn));
          if (variant->variantType == GraphType::DECODER_PREFILL) {
            ctx = m_cache_group_ctx_size[prefix];
          }
          __DEBUG("Found AR-{} CL-{} -> AR-{} CL-{}", global_arn, global_ctx, arn, ctx);
          m_cache_group_variant_map[prefix][global_variant] = {arn, ctx};
        }
        break;
      }

      if (found) break;
    }
  }
  return true;
}

// Run all validations for the model here so we can exit early
bool QnnNspModel::validateModel() {
  GENIE_TRACE();
  // Checks we will be running
  // 1a. input_ids or inputs_embeds exists in the first split
  // 1b. token_type_ids should exists in case of Bert
  // 2. logits exists in the last split
  // 3. Shapes for all named tensors are correct
  // 4. All tensors with identical names (incl kv_in/kv_out) have identical quantization params
  // Missing check : Shape of tensor between splits match up

  // Support for 16-bit KV Tensors is temporarily disabled
  // If you need this, please refer to past commits (QuaLLA <= v0.3.22)

  // Important : These variables need to be set correctly
  // m_vocab_size  - Calculated as max(logits.shape) since len()
  // m_kv_dim      - Calculated in this function before usage
  // m_ctx_size    - Provided by the user as n_ctx
  std::vector<std::tuple<std::string, std::string, std::string>> errors;

  QnnUtils::Tensor* tt;

  // default input type is token
  m_inputType = InputType::TOKENS;

  // Check 1 - input layer exists
  for (auto& [variant_spec, variant] : m_nsp_graphs.front().variants) {
    const auto& [n_tokens, ctx_size] = variant_spec;
    // Update model expectations for E2T if an inputs_embeds layer is present. marks the input Type
    if ((tt = variant->getInput("inputs_embeds")) != nullptr) {
      m_layerNames[LayerType::INPUT] = "inputs_embeds";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_embed_tokens_Gather_Gather_output_0")) != nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_embed_tokens_Gather_Gather_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_model_embed_tokens_Gather_Gather_output_0")) !=
               nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_model_embed_tokens_Gather_Gather_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_embedding_concat_Concat_Concat_output_0")) !=
               nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_embedding_concat_Concat_Concat_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    }
    if ((tt = variant->getInput(m_layerNames[LayerType::INPUT])) == nullptr) {
      errors.push_back({variant->graph_name, m_layerNames[LayerType::INPUT], "Tensor not found"});
    } else {
      input_bitwidth = tt->dtype.bw();
      checkShape(m_layerNames[LayerType::INPUT],
                 tt,
                 -1,
                 -1,
                 -1,
                 static_cast<int32_t>(input_bitwidth),
                 errors);

      if (embedding_datatype == "QNN_DATATYPE_FLOAT_32") {
        m_embeddingBufferSize = m_embd_size * sizeof(float);
      } else {
        m_embeddingBufferSize = m_embd_size * input_bitwidth;
      }

      // For embedding inputs, the expected count is multiplied by the embedding size.
      size_t expectedElementCount =
          static_cast<size_t>(n_tokens) * ((m_inputType == InputType::TOKENS) ? 1 : m_embd_size);
      if (m_layerNames[LayerType::INPUT] == "_model_embedding_concat_Concat_Concat_output_0") {
        expectedElementCount = expectedElementCount * 2;
      }
      if (tt->dims.getNumElements() != expectedElementCount)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::INPUT], "Wrong input shape"});
    }
  }

  // Check 1b - In case of BERT :-> token_type_ids
  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    for (auto& [variant_spec, variant] : m_nsp_graphs.front().variants) {
      const auto& [n_tokens, ctx_size] = variant_spec;
      if ((tt = variant->getInput(m_layerNames[LayerType::TOKEN_TYPE_IDS])) == nullptr)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::TOKEN_TYPE_IDS], "Tensor not found"});
      else {
        checkShape(m_layerNames[LayerType::TOKEN_TYPE_IDS], tt, -1, -1, -1, 4, errors);
        if (tt->dims.getNumElements() != static_cast<uint32_t>(n_tokens))
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::TOKEN_TYPE_IDS],
                            "Wrong token_type_ids shape"});
      }
    }
  }

  // Check 2 - In case of LLama :-> logits exists
  //           In case of BERT :-> pooled_output & sequence_outputs exists
  for (auto& [variant_spec, variant] : m_nsp_graphs.back().variants) {
    const auto& [n_tokens, ctx_size] = variant_spec;
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      if ((tt = variant->getOutput(m_layerNames[LayerType::POOL_OUTPUT])) == nullptr)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::POOL_OUTPUT], "Tensor not found"});
      else {
        if (tt->dims.getNumElements() != m_embd_size)
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::POOL_OUTPUT],
                            "Wrong pooled_outputs shape"});
      }
      if (!m_pooled_output) {
        if ((tt = variant->getOutput(m_layerNames[LayerType::SEQ_OUTPUT])) == nullptr)
          errors.push_back(
              {variant->graph_name, m_layerNames[LayerType::SEQ_OUTPUT], "Tensor not found"});
        else {
          if (tt->dims.getNumElements() != static_cast<uint32_t>(n_tokens) * m_embd_size)
            errors.push_back({variant->graph_name,
                              m_layerNames[LayerType::SEQ_OUTPUT],
                              "Wrong sequence_output shape"});
        }
      }
    } else {
      if(variant->variantType != GraphType::DECODER_PREFILL){
        if ((tt = variant->getOutput(m_layerNames[LayerType::OUTPUT])) == nullptr)
          errors.push_back(
              {variant->graph_name, m_layerNames[LayerType::OUTPUT], "Tensor not found"});
        else {
          m_vocab_size = (m_vocab_size == 0) ? tt->dims.getMaxDim() : m_vocab_size;
          if (tt->dims.getNumElements() != m_vocab_size &&
              tt->dims.getNumElements() != m_vocab_size * static_cast<uint32_t>(n_tokens)) {
            errors.push_back(
                {variant->graph_name, m_layerNames[LayerType::OUTPUT], "Wrong logits shape"});
          }
        }
      }
    }
  }

  // Check 3 - Shapes for all names tensors are correct
  if (m_kv_dim == -1) {  // Deduce KV$ embed_dim if not already available
    for (auto& variant : m_variant_list) {
      for (auto& [tname, tspec] : variant.output_specs) {
        if (tname.starts_with("past_key")) {
          m_kv_dim = static_cast<int32_t>(tspec.dims.width);
        }
      }

      if (m_kv_dim != -1) {
        break;
      }
    }
  }

  for (auto& variant : m_variant_list) {
    const int32_t n_tokens = variant.n_tokens;
    const int32_t ctx_size = variant.ctx_size;

    // Verify attention mask tensors
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      tt = variant.getInput(m_layerNames[LayerType::ATTN_MASK]);
      checkShape(m_layerNames[LayerType::ATTN_MASK], tt, 1, 1, ctx_size, -1, errors);
    } else {
      for (const auto& [prefix, param] : m_cache_group_params_map) {
        if (!param.attention_mask_tensor_name.empty()) {
          tt = variant.getInput(param.attention_mask_tensor_name);
          const auto& [group_arn, group_ctx] =
              m_cache_group_variant_map.at(prefix).at({n_tokens, ctx_size});
          checkShape(param.attention_mask_tensor_name, tt, 1, group_arn, group_ctx, -1, errors);
        }
      }
    }

    // Verify positional encoding tensors
    if (m_positional_encoding.type == PositionalEncoding::ROPE) {
      tt = variant.getInput(m_layerNames[LayerType::POS_SIN]);
      checkShape(m_layerNames[LayerType::POS_SIN],
                 tt,
                 1,
                 n_tokens,
                 static_cast<int32_t>(m_pos_dim),
                 -1,
                 errors);
      tt = variant.getInput(m_layerNames[LayerType::POS_COS]);
      checkShape(m_layerNames[LayerType::POS_COS],
                 tt,
                 1,
                 n_tokens,
                 static_cast<int32_t>(m_pos_dim),
                 -1,
                 errors);
    } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
      tt = variant.getInput(m_layerNames[LayerType::POS_IDS]);
      checkShape(m_layerNames[LayerType::POS_IDS], tt, 1, 1, n_tokens, -1, errors);
    } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
      tt = variant.getInput(m_layerNames[LayerType::POS_IDS]);
      checkShape(m_layerNames[LayerType::POS_IDS], tt, 1, n_tokens, ctx_size, -1, errors);
    }

    // Verify KV$ tensors
    if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
      for (const auto& [prefix, param] : m_cache_group_params_map) {
        const auto& [group_arn, group_ctx] =
            m_cache_group_variant_map.at(prefix).at({n_tokens, ctx_size});
        const int32_t past_dim =
            m_cache_group_use_scatter.at(prefix) ? group_ctx : group_ctx - group_arn;

        for (auto& [tname, tspec] : variant.input_specs) {
          if (!tname.starts_with(prefix)) continue;
          if (tname.find("key") != std::string::npos)
            checkShape(tname, &tspec, -1, m_kv_dim, past_dim, -1, errors);
          else if (tname.find("value") != std::string::npos)
            checkShape(tname, &tspec, -1, past_dim, m_kv_dim, -1, errors);
        }

        for (auto& [tname, tspec] : variant.output_specs) {
          if (!tname.starts_with(prefix)) continue;
          if (tname.find("key") != std::string::npos)
            checkShape(tname, &tspec, -1, m_kv_dim, group_arn, -1, errors);
          else if (tname.find("value") != std::string::npos)
            checkShape(tname, &tspec, -1, group_arn, m_kv_dim, -1, errors);
        }
      }
    }
  }

  // skip check in case of BERT architecture since no KV cache tensors are existing
  if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
    // Check 4 - Quantization parameter match
    std::unordered_map<std::string, QnnUtils::QuantParam> quant_params;
    for (auto& variant : m_variant_list) {
      for (auto& tensor_specs : {variant.input_specs, variant.output_specs}) {
        for (auto& [tname, tspec] : tensor_specs) {
          std::string name = tname;
          if (tname.ends_with("_in")) {  // Convert [kv_prefix]*_in to [kv_prefix]*_out
            for (const auto& [prefix, param] : m_cache_group_params_map) {
              if (tname.starts_with(prefix)) {
                name = tname.substr(0, tname.rfind("_")).append("_out");
                break;
              }
            }
          }

          if (name.compare(m_layerNames[LayerType::OUTPUT]) == 0) continue;
          if (quant_params.contains(name)) {
            if (quant_params.at(name).scale != tspec.quantParam[0].scale ||
                quant_params.at(name).offset != tspec.quantParam[0].offset) {
              errors.push_back({variant.graph_name,
                                tname,
                                "Non-identical quantization parameters found for the same tensor"});
            }
          } else {
            quant_params[tname] = {tspec.quantParam[0].scale, tspec.quantParam[0].offset};
          }
        }
      }
    }
  }

  if (errors.size() > 0) {
    QNN_ERROR("Model Validation Errors found");
    for (auto& [graph_name, tensor_name, err_msg] : errors)  // Log the list of errors
      QNN_ERROR("%s : %s - %s", graph_name.c_str(), tensor_name.c_str(), err_msg.c_str());
    QNN_ERROR("Note: Dimensions denoted by '%s' are ignored (i.e. no comparison)",
              translateDim(-1).c_str());
    QNN_ERROR("Check model i/o specs (set dump-specs=true in config) for debugging");
    State::fatal("Error validating HTP models");
    return false;
  }

  return true;
}

bool QnnNspModel::initializeKVManager() {
  GENIE_TRACE();
  if (m_lazyInitialization) return true;

  static std::map<KVManagerMode, std::string> managerModeToString = {
      {KVManagerMode::POINTER_SHIFT, "POINTER_SHIFT"},
      {KVManagerMode::SHIFT_CONCAT, "SHIFT_CONCAT"},
      {KVManagerMode::SMART_MASK, "SMART_MASK"},
      {KVManagerMode::NATIVE_KV, "NATIVE_KV"}};
  __DEBUG("Initializing with KV$ update method = {}", managerModeToString[_kv_update_method]);

  m_kvmanager = std::make_shared<KVManager>(_env, m_qnnApi.get(), m_ioTensor, m_threadpool);
  // Register supported variants
  for (const auto& graph : m_nsp_graphs) {
    for (const auto& [_, variant] : graph.variants) {
      if (variant->ctx_size != -1) {
        m_kvmanager->registerSupportedVariant(variant->n_tokens, variant->ctx_size);
      }
    }
  }

  // TODO: Select largest CL but smallest AR. We want both KV$ input and output tensors here

  // Pick largest variant/context size. This is not important for tensor mapping since
  // all buffers link to the same address anyway, but it will be important for scorer validation.
  const auto [n_tokens, ctx_size] = nsp_graph_count.rbegin()->first;
  // Initialize each cache group
  // A cache group is created for each "unique" set of KV$ Tensors
  // Unique here is defined by a difference in context size, long context modes, etc.
  // Each CacheGroup is associated with its own prefix (e.g. past_ , swa_)
  // The default CacheGroup is defined by the past_ prefix
  std::map<std::string, CacheGroup>& cache_groups = m_kvmanager->getCacheGroups();
  std::map<std::string,
           std::map<int, std::map<uint32_t, std::array<std::pair<QnnUtils::Tensor*, size_t>, 4>>>>
      group_kv_tensors;
  for (const auto& [prefix, param] : m_cache_group_params_map) {
    // Collect all KV$ tensors associated with this CacheGroup prefix
    auto& kv_map = group_kv_tensors[prefix];
    for (auto& graph : m_nsp_graphs) {
      if (!graph.variants.contains({n_tokens, ctx_size})) continue;
      GraphVariant* variant = graph(n_tokens, ctx_size);
      for (auto& [tname, tensor] : variant->output_specs) {
        if (!tname.starts_with(prefix) || tname.find("key") == std::string::npos) {
          continue;
        }

        const uint32_t index = QnnUtils::parseLayerIndex(tname);
        auto key_out_tensor  = variant->getOutput(tname);
        auto key_in_tensor   = variant->getInput(toInput(tname));
        auto val_out_tensor  = variant->getOutput(toVal(tname));
        // Get prefix for key input tensor and check if it exists in m_cache_group_ctx_size
        std::string key_in_prefix = QnnUtils::getPrefix(toInput(tname), m_cache_group_prefixes);
        size_t key_val_ctx_size   = 0;
        if (!key_in_prefix.empty() && m_cache_group_ctx_size.contains(key_in_prefix)) {
          key_val_ctx_size =
              m_cache_group_ctx_size[key_in_prefix] -
              (m_cache_group_use_scatter[key_in_prefix] ? 0 : key_out_tensor->dims.channel);
          if (!key_in_tensor &&
              variant->variantType != GraphType::DECODER_PREFILL) {  // Bert-kv models
            key_val_ctx_size = 0;
          }
        } else {
          // Use a fallback value (context size from the variant) and log a warning
          key_val_ctx_size = static_cast<size_t>(ctx_size);
          __WARN("Missing context size for key input prefix: {}. Using fallback value: {}",
                 key_in_prefix.empty() ? "empty" : key_in_prefix,
                 ctx_size);
        }
        size_t key_in_size = key_val_ctx_size * key_out_tensor->dims.batch *
                             key_out_tensor->dims.height * key_out_tensor->dims.width *
                             key_out_tensor->dims.bitwidth;

        size_t val_in_size = key_val_ctx_size * val_out_tensor->dims.batch *
                             val_out_tensor->dims.height * val_out_tensor->dims.channel *
                             val_out_tensor->dims.bitwidth;
        if (variant->variantType == GraphType::DECODER_PREFILL) {
          const auto [min_variant, min_ctx_size] = nsp_graph_count.begin()->first;
          GraphVariant* variantDecoder =
              graph(min_variant,
                    min_ctx_size);  // Use Decoder Graph variant for initializing KV IN tensors
          kv_map[graph.idx()][index] = std::array<std::pair<QnnUtils::Tensor*, size_t>, 4>{
              std::make_pair(variantDecoder->getInput(toInput(tname)), key_in_size),
              std::make_pair(&tensor, tensor_alloc_info[tname].second),
              std::make_pair(variantDecoder->getInput(toVal(toInput(tname))), val_in_size),
              std::make_pair(variant->getOutput(toVal(tname)),
                             tensor_alloc_info[toVal(tname)].second)};
        } else {
          kv_map[graph.idx()][index] = std::array<std::pair<QnnUtils::Tensor*, size_t>, 4>{
              std::make_pair(variant->getInput(toInput(tname)), key_in_size),
              std::make_pair(&tensor, tensor_alloc_info[tname].second),
              std::make_pair(variant->getInput(toVal(toInput(tname))), val_in_size),
              std::make_pair(variant->getOutput(toVal(tname)),
                             tensor_alloc_info[toVal(tname)].second)};
        }

        if (variant->variantType != GraphType::DECODER_PREFILL) {
          if (kv_map.at(graph.idx()).at(index)[3].first == nullptr) {
            uint16_t layer_idx = index >> 16, head_idx = index & 0xffff;
            State::error(fmt::format("Error in layer {} head {}. ", layer_idx, head_idx) +
                         fmt::format("Found Key {} but no Value {}", tname, toVal(tname)));
            return false;
          }
        }
      }
    }
    bool use_scatter = false;
    if (kv_map.empty()) {
      if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
        State::error(fmt::format("Invalid cache-group prefix detected: {}", prefix));
        return false;
      }
    } else {
      use_scatter = m_cache_group_use_scatter.at(prefix);
    }
    cache_groups.emplace(prefix, CacheGroup(_env, prefix, use_scatter, param.longcontext_params));
  }
  // Register KVTensors into each CacheGroup
  for (auto& [prefix, cache_group] : cache_groups) {
    cache_group.context_manager->cache_group = &cache_group;

    auto& kv_map = group_kv_tensors[prefix];
    cache_group.registerTensors(kv_map);

    cache_group.m_variant_map = m_cache_group_variant_map.at(prefix);

    if (cache_group.context_manager->params.mode != LongContextParams::KEYDIFF) continue;

    // Get buffers for anchor input, anchor output and score tensors
    // Get the allocation information for Anchor input buffer and Key Cache Buffer
    std::map<uint32_t, std::array<std::tuple<int, size_t>, 2>> scorer_allocs;
    std::map<uint32_t, std::array<QnnUtils::Tensor*, 2>> anchor_tensors;
    for (auto& graph : m_nsp_graphs) {
      GraphVariant* variant = graph(n_tokens, ctx_size);
      for (auto& [tname, tensor] : variant->input_specs) {
        if (!tname.starts_with(m_layerNames.at(LayerType::ANCHOR))) continue;
        const uint32_t index  = QnnUtils::parseLayerIndex(tname);
        anchor_tensors[index] = {&tensor, variant->getOutput(toOutput(tname))};
        scorer_allocs[index]  = {graph.tensor_alloc_info->at(tname),
                                 graph.tensor_alloc_info->at(QNN_TENSOR_GET_NAME(
                                    kv_map.at(graph.idx()).at(index)[1].first->tensor))};
      }
    }
    // Initialize all tensors for the scorer model (anchor/keys/scores)
    // Also add the score buffer pointer associated with each
    std::string scorer_path =
        (model_basedir / fs::path(cache_group.context_manager->params.scoring_network)).string();
    __DEBUG("Initializing KeyDiff Scorer {}", scorer_path);
    std::map<uint32_t, uint8_t*> score_memptr;
    if (!m_qnnApi->initializeScorer(scorer_path,
                                    scorer_allocs,
                                    score_memptr,
                                    static_cast<size_t>(ctx_size),
                                    m_expectedDataFormat)) {
      State::error("Failed to initialize scorer");
      return false;
    }
    __DEBUG("cache group = {:p} keydiff.group={:p}",
            fmt::ptr(&cache_group),
            fmt::ptr(cache_group.context_manager->cache_group));

    __DEBUG("anchor_tensors = [");
    for (auto& [index, anchor_io] : anchor_tensors)
      __DEBUG("\t{}: [{}, {}, {}] {}",
              index,
              fmt::ptr(anchor_io[0]),
              fmt::ptr(anchor_io[1]),
              fmt::ptr(score_memptr.at(index)),
              cache_group.m_tensor_index.contains(index));
    __DEBUG("]");
    auto keydiff = dynamic_cast<KeyDiff*>(cache_group.context_manager.get());
    keydiff->registerKeydiffBuffers(anchor_tensors, score_memptr);
    __DEBUG("Completed registerKeydiffBuffers");

    for (auto& [index, t] : cache_group.m_tensor_index) {
      __DEBUG("\t{}:[anchor in={:p} out={:p} score={:p}]",
              index,
              fmt::ptr(t->anchor_tensor_in),
              fmt::ptr(t->anchor_tensor_out),
              fmt::ptr(t->scores));
    }
  }

  if (_kv_update_method == KVManagerMode::NATIVE_KV) {
    std::map<std::pair<int32_t, int32_t>, bool> isKvOutputNativeFormat;
    for (auto& [prefix, cache_group] : cache_groups) {
      bool found_decoder_layer = false;
      for (auto& select_graph : m_nsp_graphs) {
        if (found_decoder_layer) break;
        for (const auto& [key, value] : nsp_graph_count) {
          int32_t var           = key.first;
          int32_t ctx           = key.second;
          GraphVariant* variant = select_graph(var, ctx);
          if (variant->variantType != GraphType::DECODER &&
              variant->variantType != GraphType::DECODER_PREFILL)
            break;
          found_decoder_layer = true;
          for (auto& [mtname, mtensor] : variant->output_specs) {
            if (mtname.starts_with(prefix) && QnnUtils::isKVTensor(mtname)) {
              isKvOutputNativeFormat[key] =
                  mtensor.tensor->v1.dataFormat == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;
              if (!isKvOutputNativeFormat[key]) {
                __WARN("The graph {}'s KVCache has Native input and FlatBuffer output",
                       variant->graph_name);
              }
              break;
            }
          }
        }
      }
      cache_group.registerKvOutputNativeFormat(isKvOutputNativeFormat);
    }
  }
  m_kvmanager->initComplete(m_ctx_size, m_default_group);
  if (m_kvmanager->failed()) {
    State::fatal(m_kvmanager->error());
    return false;
  }
  m_kvmanager->dispatchUpdate(0);
  if (m_kvmanager->failed()) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  // Detect which variants have logits outputs
  std::set<std::pair<int32_t, int32_t>> logit_containing_variants;
  QnnNspGraph& last_split = m_nsp_graphs.back();
  for (auto& [variant_spec, graph_variant] : last_split.variants) {
    QnnUtils::Tensor* logit_tensor = graph_variant->getOutput(m_layerNames[LayerType::OUTPUT]);
    if (logit_tensor != nullptr) {
      logit_containing_variants.insert(variant_spec);
    }
  }
  m_kvmanager->registerLogitVariants(logit_containing_variants);

  return true;
}

inline bool QnnNspModel::updateTensorPointer(GraphVariant& variant,
                                             std::string& key,
                                             QnnUtils::Tensor*& t) {
  QnnUtils::Tensor* tensor_ptr = variant.getInput(key);
  if (tensor_ptr == nullptr) return true;
  if (t == nullptr) t = tensor_ptr;
  if (getBuffer(t) == getBuffer(tensor_ptr)) return true;

  __ERROR("{} has different addresses: {} vs {}",
          key,
          static_cast<void*>(t),
          static_cast<void*>(tensor_ptr));
  return false;
}

bool QnnNspModel::initializeTensorPointers() {
  GENIE_TRACE();
  // Ideally this needs to be done for all sets of AR-n available, e.g. for AR-1 and AR-1024
  if (m_lazyInitialization) return true;
  bool status = true;
  for (auto& variant : m_variant_list) {
    status &= updateTensorPointer(variant, m_layerNames[LayerType::INPUT], t_input_ids);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::ATTN_MASK], t_attn_mask);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_SIN], t_position_ids_sin);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_COS], t_position_ids_cos);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_IDS], t_position_ids);
    status &=
        updateTensorPointer(variant, m_layerNames[LayerType::TOKEN_TYPE_IDS], t_token_type_ids);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::VALID_MASK], t_valid_mask);
  }
  if (!status) __ERROR("qnn-htp: Error in setting up named tensor pointers.");

  // Find tensors for each group, iff it's been provided via the user config
  for (auto& [prefix, param] : m_cache_group_params_map) {
    QnnUtils::Tensor* group_attn_mask{nullptr};
    QnnUtils::Tensor* group_cache_index{nullptr};
    for (auto& variant : m_variant_list) {
      if (!param.attention_mask_tensor_name.empty()) {
        status &= updateTensorPointer(variant, param.attention_mask_tensor_name, group_attn_mask);
      }
      if (!param.cache_index_tensor_name.empty()) {
        status &= updateTensorPointer(variant, param.cache_index_tensor_name, group_cache_index);
      }
    }

    if (!param.attention_mask_tensor_name.empty()) {
      if (!group_attn_mask) {
        status = false;
        __ERROR(
            "Couldn't find attn mask {} for group {}", param.attention_mask_tensor_name, prefix);
      } else {
        m_group_attn_mask[prefix] = group_attn_mask;
      }
    }
    if (!param.cache_index_tensor_name.empty()) {
      if (!group_cache_index) {
        status = false;
        __ERROR("Couldn't find cache-index {} for group {}", param.cache_index_tensor_name, prefix);
      } else {
        m_group_cache_index[prefix] = group_cache_index;
      }
    }
  }

  status &= !(!t_input_ids || !t_attn_mask);
  if (!t_input_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::INPUT]);
  if (!t_attn_mask) __ERROR("Tensor not found: {}", m_layerNames[LayerType::ATTN_MASK]);

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {  // This input only valid for
                                                                    // Encoder only model like bert.
    status &= !(!t_token_type_ids);
    if (!t_token_type_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::TOKEN_TYPE_IDS]);
  }

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    status &= !(!t_position_ids_sin || !t_position_ids_cos);
    if (!t_position_ids_sin) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_SIN]);
    if (!t_position_ids_cos) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_COS]);
  } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
    status &= !(!t_position_ids);
    if (!t_position_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_IDS]);
  } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
    status &= !(!t_position_ids);
    if (!t_position_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_IDS]);
  } else {
    __ERROR("Unknown Rope Type found for tensor: {}", m_layerNames[LayerType::POS_IDS]);
  }

  // Detect activation bitwidth
  if (status) {
    // Check Input-> Input_ID or Input_Embed
    d_input = t_input_ids->dtype;
    if (!isSupportedActivation(d_input)) {
      __ERROR("Input Tensor: {} as unsupported activation type {}",
              m_layerNames[LayerType::INPUT],
              d_input.str());
      status = false;
    }
    // Check Attention Mask
    d_attn_map = t_attn_mask->dtype;
    if (!isSupportedActivation(d_attn_map)) {
      __ERROR("attention_mask has unsupported type {}", d_attn_map.str());
      status = false;
    }

    uint32_t attn_bitwidth = d_attn_map.bw();
    bool attn_quantized    = (d_attn_map.type() != 2);
    if (attn_quantized) {
      // Support uint8, uint16 and uint32
      if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
        m_attention_positive_value.u32 = 1;  // This sets u8=u16=u32=1
      } else {
        m_attention_positive_value.u32 = 0xffffffff;  // This sets u8=0xff u16=0xffff
      }
      m_attention_negative_value.u32 = 0;  // This sets u8=u16=u32=0
    } else {
      // Support float16 or float32
      m_attention_positive_value.u32 = 0;  // Set u16=u32=0 for fp16 or fp32
      if (attn_bitwidth == 1) {            // float8 is not currently supported
        status = false;
      } else if (attn_bitwidth == 2) {
        m_attention_negative_value.u16 = fp16_ieee_from_fp32_value(-1000.0f);
      } else if (attn_bitwidth == 4) {
        float value = -1000.0f;
        std::memcpy(&m_attention_negative_value.u32, &value, sizeof(float));
      }
    }

    // For Encoder only model, Check for Token_type_ids
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      d_token_type = t_token_type_ids->dtype;
      if (!isSupportedActivation(d_token_type)) {
        __ERROR("token_type_ids has unsupported type {}", d_token_type.str());
        status = false;
      }
    }

    // For Position_IDs check data bitwidth
    if (m_positional_encoding.type == PositionalEncoding::ROPE) {
      d_pos = t_position_ids_sin->dtype;
    }
    else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
      d_pos = t_position_ids->dtype;
    }
    else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
      d_pos = t_position_ids->dtype;
    }

    if (((m_positional_encoding.type == PositionalEncoding::ABSOLUTE ||
          m_positional_encoding.type == PositionalEncoding::ALIBI) &&
         d_pos != QNN_DATATYPE_INT_32) ||
        (m_positional_encoding.type == PositionalEncoding::ROPE &&
         !isSupportedActivation(d_pos))) {
      __ERROR("position encoding tensor has unsupported type {}", d_pos.str());
      status = false;
    }

    if (t_valid_mask != nullptr && t_valid_mask->dtype != QNN_DATATYPE_UFIXED_POINT_16) {
      __ERROR("Valid mask tensor has unsupported type {}", t_valid_mask->dtype.str());
      status = false;
    }

    __DEBUG("qnn-htp datatypes: d_input {} d_attn_map {} d_pos {}",
            d_input.str(),
            d_attn_map.str(),
            d_pos.str());

    if (!status) __ERROR("Only 8-bit, 16-bit and 32-bit activations are supported");
  }

  return status;
}

template <typename DType>
void QnnNspModel::setupAttentionMask(const InferenceStep& step, AttentionMask& attention_mask) {
  GENIE_TRACE();
  DType pos_val, neg_val;
  if constexpr (std::is_same_v<DType, uint8_t>) {
    pos_val = m_attention_positive_value.u8;
    neg_val = m_attention_negative_value.u8;
  } else if constexpr (std::is_same_v<DType, uint16_t>) {
    pos_val = m_attention_positive_value.u16;
    neg_val = m_attention_negative_value.u16;
  } else {
    pos_val = m_attention_positive_value.u32;
    neg_val = m_attention_negative_value.u32;
  }

  const size_t variant    = static_cast<size_t>(step.variant);
  const size_t ctx_size   = static_cast<size_t>(step.ctx_size);
  const size_t n_past     = static_cast<size_t>(step.n_past);
  const size_t n_valid_kv = static_cast<size_t>(step.n_valid_kv);
  const size_t n_process  = static_cast<size_t>(step.n_process);
  const size_t past_idx   = static_cast<size_t>(step.past_idx);
  const size_t new_idx    = static_cast<size_t>(step.new_idx);

  DType* attn_buffer = reinterpret_cast<DType*>(getBuffer(t_attn_mask));

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    size_t n_valid = n_valid_kv + n_process;
    size_t offset  = (variant == ctx_size) ? ctx_size - n_valid : 0;
    std::fill_n(attn_buffer, ctx_size, neg_val);
    std::fill_n(attn_buffer + offset, n_valid, pos_val);
    return;
  }

  // Clear entire attention buffer
  std::fill_n(attn_buffer, variant * ctx_size, neg_val);

  // Fill the attention mask row-by-row
  for (size_t i = 0; i < n_process; i++) {
    attention_mask.fillAttentionRow<DType>(std::span<DType>(&attn_buffer[i * ctx_size], ctx_size),
                                           i,
                                           n_past,
                                           n_valid_kv,
                                           past_idx,
                                           new_idx,
                                           pos_val);
  }

  // Handle attention masks for non-default cache groups
  for (auto& [prefix, param] : m_cache_group_params_map) {
    if (prefix == m_default_group || !m_group_attn_mask.contains(prefix)) {
      continue;
    }

    if (param.longcontext_params.mode != LongContextParams::SLIDING_WINDOW) {
      __ERROR("CacheGroup-specific attention mask only supported for SWA Cache groups");
      continue;
    }

    // Step 1: Get the tensor for this groups attention mask
    // Step 2: Parse the indexes based on the group's current state
    // Step 3: Construct the group attention mask based on the global attention mask

    // Step 1
    DType* group_attn_buffer = static_cast<DType*>(getBuffer(m_group_attn_mask[prefix]));

    // Step 2
    CacheGroup& group = m_kvmanager->getCacheGroups().at(prefix);
    const std::vector<std::pair<int32_t, size_t>>& gather_indexes =
        group.context_manager->translateAttentionMask(step);
    __DEBUG("SWA Gather index = {}", gather_indexes);

    // Calculate an exclusion zone, so we don't attend more tokens than our max_attention_span
    const int32_t max_attention_span = param.longcontext_params.window_size;
    const int32_t exclusion_start = std::max(step.n_valid_kv - group.m_n_valid_kv, static_cast<int32_t>(_size_to_skip_kv_prefix));
    const int32_t exclusion_end = step.n_valid_kv - max_attention_span + 1;

    // Step 3
    std::fill_n(group_attn_buffer, group.m_cur_variant * group.m_cur_ctx, neg_val);
    const auto& position_ids = attention_mask.getPositionIds(
      n_past - attention_mask.get_n_past(),
      n_process,
      variant
    );
    for (size_t i = 0; i < n_process; i++) {
      size_t row_offset = 0;
      const int32_t row_exclusion_end = exclusion_end + position_ids[i] - position_ids[0];

      // Check if exclusion zone should be disabled
      const bool disable_exclusion = (row_exclusion_end <= exclusion_start || max_attention_span <= 0);

      // If exclusion zone is invalid or max_attention_span is 0, set it beyond the context size to disable exclusion
      const size_t exclude_start_idx = static_cast<size_t>(disable_exclusion ? step.ctx_size : exclusion_start);
      const size_t exclude_end_idx = static_cast<size_t>(disable_exclusion ? step.ctx_size : row_exclusion_end);

      for (const auto& [offset, count] : gather_indexes) {
        if (offset < 0) {
          row_offset += count;
          continue;
        }

        const size_t local_ctx = static_cast<size_t>(group.m_cur_ctx);
        const size_t global_start_idx = static_cast<size_t>(offset);
        const size_t global_end_idx = global_start_idx + count;

        // Check for overlap with exclusion zone
        const bool has_overlap = !(exclude_end_idx <= global_start_idx ||
                                   global_end_idx <= exclude_start_idx);

        if (!has_overlap) {
          // No overlap - copy entire range
          std::memcpy(&group_attn_buffer[i * local_ctx + row_offset],
                      &attn_buffer[i * ctx_size + global_start_idx],
                      count * sizeof(DType));
        } else {
          // Handle overlap by copying before and after exclusion zone
          
          // Copy segment before exclusion zone
          if (global_start_idx < exclude_start_idx) {
            const size_t before_count = exclude_start_idx - global_start_idx;
            std::memcpy(&group_attn_buffer[i * local_ctx + row_offset],
                        &attn_buffer[i * ctx_size + global_start_idx],
                        before_count * sizeof(DType));
          }
          
          // Copy segment after exclusion zone
          if (exclude_end_idx < global_end_idx) {
            const size_t after_count = global_end_idx - exclude_end_idx;
            const size_t dst_offset = row_offset + (exclude_end_idx - global_start_idx);

            std::memcpy(&group_attn_buffer[i * local_ctx + dst_offset],
                        &attn_buffer[i * ctx_size + exclude_end_idx],
                        after_count * sizeof(DType));
          }
        }

        row_offset += count;
      }
    }
  }
}

template <typename DType>
bool QnnNspModel::setupAlibiPositionEmbedding(const InferenceStep& step) {
  DType* alibi_buffer = reinterpret_cast<DType*>(getBuffer(t_position_ids));
  const DType pad_val = static_cast<DType>(step.ctx_size);

  // Clear alibi buffer
  std::fill_n(alibi_buffer, step.variant * step.ctx_size, pad_val);

  // Detect start of past tokens and new tokens based on ctx_size and n_tokens (variant)
  DType* alibi_past = &alibi_buffer[step.past_idx];
  DType* alibi_new  = &alibi_buffer[step.new_idx];

  // Fill alibi positions from [-n_past-i, -i) and [-i, 0]
  for (int i = 0; i < step.n_process; i++) {
    std::iota(std::reverse_iterator<DType*>(alibi_past + step.n_past),
              std::reverse_iterator<DType*>(alibi_past),
              i + 1);  // Fill past tokens
    std::iota(std::reverse_iterator<DType*>(alibi_new + i + 1),
              std::reverse_iterator<DType*>(alibi_new),
              0);  // Fill new tokens

    alibi_past += step.ctx_size;  // Update pointers to next row
    alibi_new += step.ctx_size;
  }

  return true;
}

bool QnnNspModel::setupInputEmbeddings(const InferenceStep& step,
                                       bool /*pad_left*/,
                                       const std::vector<uint8_t>& eagle_embed,
                                       const uint16_t* eagle_feature_in,
                                       const std::vector<int32_t>& selected,
                                       uint32_t start_idx,
                                       uint32_t embed_in_idx,
                                       bool post_update) {
  size_t in_buf_offset           = 0;
  uint16_t* embed_ptr            = reinterpret_cast<uint16_t*>(getBuffer(t_input_ids));
  const uint16_t* eagle_embed_in = reinterpret_cast<const uint16_t*>(eagle_embed.data());
  uint16_t embedDraft            = getEmbeddingBufferSize();
  size_t count                   = eagle_embed.size() / embedDraft;
  size_t offset_len              = embedDraft;
  size_t feature_len             = embedDraft;
  size_t embed_len               = embedDraft;
  uint16_t increm                = m_embedding_length;
  embed_ptr += start_idx * offset_len;

  uint16_t* feature_in_ptr = embed_ptr + embed_len / sizeof(uint16_t);
  void* feature_in_buffer  = nullptr;

  getIOBufferByName(draftFeatureNameIn, feature_in_buffer, false);
  bool isDualHead          = (feature_in_buffer != nullptr);
  size_t offset_divide_len = 1;
  // Using dual head input instead of concat if found draftFeatureNameIn in draft model
  if (isDualHead) {
    feature_in_ptr = reinterpret_cast<uint16_t*>(feature_in_buffer);
    feature_in_ptr += start_idx * feature_len / sizeof(uint16_t);
    embed_ptr += start_idx * offset_len / sizeof(uint16_t);
    offset_divide_len = sizeof(uint16_t);
  }
  if (selected.size() == 0) {
    if (eagle_extra_feature == nullptr) {
      eagle_extra_feature = reinterpret_cast<uint16_t*>(malloc(feature_len));
      // clear the extra feature buffer
      std::memset(eagle_extra_feature, 0, feature_len);
    } else {
      const uint16_t* embed_data = (eagle_embed_in);
      // concat the embedding and feature vector data
      std::memcpy(embed_ptr, embed_data, embed_len);
      std::memcpy(feature_in_ptr, eagle_extra_feature, feature_len);
    }
    embed_ptr += offset_len;
    feature_in_ptr += offset_len / offset_divide_len;

    for (size_t i = 1; i < static_cast<size_t>(step.variant); i++) {
      // update the pointer to next token embedding data
      const uint16_t* embed_data = eagle_embed_in + (i * increm);
      // feature data of one before token to copied
      const uint16_t* feature_data = eagle_feature_in + (i - 1 - in_buf_offset) * feature_len / 2;
      // concat the embedding and feature vector data
      std::memcpy(embed_ptr, embed_data, embed_len);
      std::memcpy(feature_in_ptr + embed_len / sizeof(uint16_t), feature_data, feature_len);
      embed_ptr += offset_len;
      feature_in_ptr += offset_len / offset_divide_len;
    }
    std::memcpy(eagle_extra_feature, eagle_feature_in + step.n_process - 1, feature_len);
  } else {
    if (selected.size() != count && selected.size() != count + 1) {
      __ERROR("setupInputEmbeddings ERROR: wrong selected vector size");
      return false;
    }
    if (eagle_extra_feature == nullptr) {
      eagle_extra_feature = reinterpret_cast<uint16_t*>(malloc(feature_len));
      std::memset(eagle_extra_feature, 0, feature_len);
    }
    const uint16_t* embed_data   = nullptr;
    const uint16_t* feature_data = nullptr;

    size_t copy_buffer_size = std::min(embed_in_idx + static_cast<size_t>(step.variant), count);
    for (uint32_t j = embed_in_idx + start_idx; j < copy_buffer_size; j++) {
      // update the pointer to next token embedding data
      embed_data = (eagle_embed_in + j * increm);
      if (selected[j] >= 0) {
        // Feature data to be copied as per selected idx,each sequence only see the parent and
        // and its predecessor
        feature_data = eagle_feature_in +
                       (static_cast<size_t>(selected[j]) * (feature_len / sizeof(uint16_t)));
      } else {
        // if selection id is -1 used the last iteration last feature vector.
        feature_data = eagle_extra_feature;
      }
      // concat the embedding and feature vector data
      std::memcpy(embed_ptr, embed_data, embed_len);
      std::memcpy(feature_in_ptr, feature_data, feature_len);
      embed_ptr += offset_len / offset_divide_len;
      feature_in_ptr += offset_len / offset_divide_len;
    }

    if (!post_update) {
      size_t feature_end_idx =
          (copy_buffer_size == embed_in_idx + static_cast<uint32_t>(step.variant))
              ? copy_buffer_size - 1
              : copy_buffer_size;

      feature_data = eagle_feature_in + ((feature_end_idx - embed_in_idx - in_buf_offset) *
                                         feature_len / sizeof(uint16_t));

      // store the extra feature buffer to be used in next iteration, if selection if is -1
      std::memcpy(eagle_extra_feature, feature_data, feature_len);
    }
  }
  return true;
}

bool QnnNspModel::setupInput(const InferenceStep& step,
                             uint32_t start,
                             const std::vector<int32_t>& tokens,
                             std::vector<uint8_t>& embeddings,
                             const uint16_t* featureVector,
                             const std::vector<int32_t>& selected,
                             const uint32_t start_idx,
                             const bool post_update,
                             AttentionMask& attention_mask) {
  GENIE_TRACE();
  const size_t variant   = static_cast<size_t>(step.variant);
  const size_t ctx_size  = static_cast<size_t>(step.ctx_size);
  const size_t n_past    = static_cast<size_t>(step.n_past);
  const size_t n_process = static_cast<size_t>(step.n_process);

  if (!tokens.empty()) {
    // Setup input id tensor
    uint32_t* input_id_buffer = reinterpret_cast<uint32_t*>(getBuffer(t_input_ids));
    std::fill_n(input_id_buffer, variant, static_cast<uint32_t>(m_pad_token));
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      const size_t pad_offset = (variant == ctx_size) ? variant - n_process : 0;
      std::memcpy(&input_id_buffer[pad_offset], &tokens[start], n_process * sizeof(uint32_t));
    } else if (variant == ctx_size) {
      // Special handling for AR-c models. All past tokens must be re-processed
      const size_t n_history = token_history.size();
      std::memcpy(input_id_buffer, token_history.data(), n_history * sizeof(uint32_t));
      std::memcpy(
          &input_id_buffer[n_history], &tokens[start], (n_process - n_history) * sizeof(uint32_t));
    } else {
      // For normal cases (variant < ctx_size), tokens are processed normally
      std::memcpy(input_id_buffer, &tokens[start], n_process * sizeof(uint32_t));
    }
  } else if (!embeddings.empty() &&
             featureVector == nullptr) {  // Quantize and fill, don't make double copy

    if (embedding_datatype == "QNN_DATATYPE_FLOAT_32") {
      // First flush the buffer with eos token embedding
      float* embeddingSrc = reinterpret_cast<float*>(m_eosEmbedding.data());
      for (size_t i = n_process; i < variant; i++) {
        quantizeInput(
            embeddingSrc, i * static_cast<size_t>(m_embd_size), static_cast<size_t>(m_embd_size));
      }

      // Quantize the data input vector
      embeddingSrc = reinterpret_cast<float*>(embeddings.data());
      quantizeInput(&embeddingSrc[start * static_cast<size_t>(m_embd_size)],
                    0,
                    n_process * static_cast<size_t>(m_embd_size));
    } else {
      // Size of the buffer for one embedding vector.
      const size_t embedBufSize = m_embeddingBufferSize;
      // First flush the buffer with eos token embedding
      uint8_t* embeddingSrc = static_cast<uint8_t*>(m_eosEmbedding.data());
      if (!embeddingSrc) {
        __ERROR("setupInput : EOS embedding data is NULL.");
        return false;
      }
      for (uint32_t i = n_process; i < variant; i++) {
        std::copy(embeddingSrc,
                  embeddingSrc + embedBufSize,
                  reinterpret_cast<uint8_t*>(getBuffer(t_input_ids)) + i * embedBufSize);
      }

      // Copy the data input vector
      embeddingSrc = static_cast<uint8_t*>(embeddings.data()) + (start * embedBufSize);
      std::copy(embeddingSrc,
                embeddingSrc + (n_process * embedBufSize),
                reinterpret_cast<uint8_t*>(getBuffer(t_input_ids)));
    }
  } else if (!embeddings.empty() && featureVector != nullptr) {
    setupInputEmbeddings(
        step, false, embeddings, featureVector, selected, start_idx, start, post_update);
  }

  int32_t cache_index_boundary =
      step.ctx_size -
      static_cast<int32_t>(std::ceil(static_cast<double>(step.variant) / 32.0) * 32);

  if (_kv_update_method == KVManagerMode::NATIVE_KV && step.new_idx > cache_index_boundary) {
    State::error(fmt::format("Error: cache_index {} cannot be greater than {} in native mode.",
                             step.new_idx,
                             cache_index_boundary));
    return false;
  }
  // Set up the input scatter index as new_idx (i.e. the index where new KV$ is stored)
  for (auto& [prefix, group_index_tensor] : m_group_cache_index) {
    CacheGroup& group        = m_kvmanager->getCacheGroups().at(prefix);
    InferenceStep group_step = group.translateInferenceStep(step);

    uint32_t* group_index_buffer = reinterpret_cast<uint32_t*>(getBuffer(group_index_tensor));
    std::iota(group_index_buffer,
              group_index_buffer + group_index_tensor->dims.getNumElements(),
              group_step.new_idx);
  }

  if (t_valid_mask != nullptr) {  // Set up a valid mask. Assumes u16 datatype (from validateModel)
    // Quantize mask value to u16
    const auto [scale, offset] = t_valid_mask->quantParam[0];
    uint16_t mask_val = QnnUtils::quantize<double, uint16_t>(1.0 / n_process, offset, scale);

    bool hasSpeculativeTokens = false;
    if (!tokens.empty()) {
      for (int32_t token : tokens) {
        if (token >= static_cast<int32_t>(m_vocab_size)) {
          hasSpeculativeTokens = true;
          break;
        }
      }
    }

    const size_t n_masked = hasSpeculativeTokens ? 1 : n_process;
    // Setup the buffer
    uint16_t* mask_buffer = reinterpret_cast<uint16_t*>(getBuffer(t_valid_mask));
    std::fill_n(mask_buffer, n_masked, mask_val);
    std::memset(&mask_buffer[n_masked], 0, (variant - n_masked) * sizeof(uint16_t));
  }

  // Setup the attention mask correctly
  if (d_attn_map.bw() == 1) {
    setupAttentionMask<uint8_t>(step, attention_mask);
  } else if (d_attn_map.bw() == 2) {
    setupAttentionMask<uint16_t>(step, attention_mask);
  } else if (d_attn_map.bw() == 4) {
    setupAttentionMask<uint32_t>(step, attention_mask);
  }

  // Setup token type IDs
  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    // BERT Specific
    uint32_t* token_type_id_buffer = reinterpret_cast<uint32_t*>(getBuffer(t_token_type_ids));
    std::memset(token_type_id_buffer, 0, variant * sizeof(uint32_t));
  }

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    // Simple RoPE position ID setup
    const auto& position_ids = attention_mask.getPositionIds(start, n_process, variant);
    // __DEBUG("Position IDs = {}", position_ids);

    // TODO: Compile position_ids to translate from [0,1,2,3,4,0,0,...,0] -> [(0,5), (0,1),...]
    // This is to batch the memory copy calls together, which is more optimal (theoretically)
    uint8_t* cos_buffer    = reinterpret_cast<uint8_t*>(getBuffer(t_position_ids_cos));
    uint8_t* sin_buffer    = reinterpret_cast<uint8_t*>(getBuffer(t_position_ids_sin));
    const size_t rope_size = m_pos_dim * d_pos.bw();
    for (uint32_t i = 0; i < variant; i++) {
      const size_t src_offset = static_cast<size_t>(position_ids[i]) * rope_size;
      const size_t dst_offset = i * rope_size;
      std::memcpy(
          &sin_buffer[dst_offset], reinterpret_cast<uint8_t*>(rope_sin) + src_offset, rope_size);
      std::memcpy(
          &cos_buffer[dst_offset], reinterpret_cast<uint8_t*>(rope_cos) + src_offset, rope_size);
    }
  } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
    uint32_t* position_id_buffer = reinterpret_cast<uint32_t*>(getBuffer(t_position_ids));
    std::memset(position_id_buffer, 0, variant * sizeof(uint32_t));

    // Fill up position_ids buffer
    size_t pad_offset =
        (m_modelArchitectureType == ModelArchitectureType::ENCODER) ? variant - n_process : 0;
    uint32_t* pos_id_start = position_id_buffer + pad_offset;
    uint32_t* pos_id_end   = pos_id_start + n_process;
    std::iota(pos_id_start, pos_id_end, n_past);
  } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
    setupAlibiPositionEmbedding<int32_t>(step);
  }
  return true;
}

inline void QnnNspModel::syncDrafTargetPrefill(bool isDraft, bool isReset) {
  if (_counter == nullptr) {
    return;
  }
  if (isReset == false) {
    if (isDraft == true) {
      while (_counter != nullptr && *_counter == 0) {
      }
    } else {
      while (_counter != nullptr && *_counter != 0) {
      }
    }
  } else {
    if (isDraft) {
      if (_counter != nullptr) {
        *_counter = 0;
      }
    } else {
      if (_counter != nullptr) {
        *_counter = 1;
      }
    }
  }
}

size_t QnnNspModel::runInference(const std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embedding,
                                 const uint16_t* featureVector,
                                 const std::vector<int32_t>& selected,
                                 uint32_t start_idx,
                                 bool post_update,
                                 const std::vector<int32_t>& attention_map,
                                 std::vector<float>& output,
                                 bool output_all) {
  GENIE_TRACE();
  qualla::Timer start;
  __TRACE("runInference logits_all={} tokens={} featureVector {}",
          output_all,
          tokens,
          reinterpret_cast<uintptr_t>(featureVector));

  bool draft = false;
  if (featureVector != nullptr) {
    draft = true;
  }

  if ((tokens.size() == 0) && (embedding.size() == 0)) return 0;

  size_t embedBufSize   = m_embeddingBufferSize;
  size_t embeddingCount = embedding.size() / embedBufSize;

  // Disable token_history (required for AR-c models) if embedding input is processed
  if (embeddingCount > 0) {
    token_history_enabled = false;
  }

  // Construct an attention mask processor to simplify handling of complex masks
  const size_t n_inputs = tokens.size() + embeddingCount;
  AttentionMask attention_mask(attention_map,
                               static_cast<size_t>(m_kvmanager->n_past()),
                               static_cast<size_t>(m_kvmanager->n_valid_kv()),
                               n_inputs,
                               _offset_to_apply_kv_prefix,
                               _size_to_skip_kv_prefix);

  if (attention_map.size() > n_inputs && isLongContextEnabled()) {
    State::fatal("LongContext has not been enabled for this dialog");
    return false;
  }

  // Create a strategy to run the inference
  if (!m_kvmanager->prepareInferenceStrategy(static_cast<int32_t>(n_inputs))) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  // user choice overwrites the default behaviour in case of Embedding models
  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    output_all = !m_pooled_output;
  }

  size_t output_size = output_all ? n_inputs : 1;  // actual number of logits
  output.resize(output_size * ((m_modelArchitectureType == ModelArchitectureType::ENCODER)
                                   ? m_embd_size
                                   : m_vocab_size));
  __TRACE("runInference output ={}", output.size());

  // Loop over each planned iteration in the inference strategy
  InferenceStep step;
  uint32_t n_processed = 0;  // Number of total tokens processed so far

  while (m_kvmanager->nextInferenceStep(step)) {
    if (m_pause && n_processed != 0 && n_processed != 1) {
      m_pause = false;
      return n_processed;
    }

    __DEBUG("Inference step: {}", step.str());
    syncDrafTargetPrefill(draft, false);
    if (!setupInput(step,
                    n_processed,
                    tokens,
                    embedding,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_mask))
      return false;

    for (auto& nsp_graph : m_nsp_graphs) {
      const int graph_idx = nsp_graph.idx();

      if (nsp_graph.m_graphType == GraphType::LMHEAD && (!output_all) &&
          !m_kvmanager->isFinalInferenceStep()) {  // Must meet all the criteria to skip LMHEAD
        continue;
      }

      if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }

      if (!nsp_graph.execute(
              step.variant, step.ctx_size, m_inference_count, graph_switching, lazy_lora)) {
        fatal(fmt::format("Failed to execute graph {}", graph_idx));
        return false;
      }

      if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }
    }
    m_kvmanager->completeInferenceStep();

    if (m_modelArchitectureType != ModelArchitectureType::ENCODER && output_all)
      getDequantLogits(std::span(&output[n_processed * m_vocab_size],
                                 static_cast<uint32_t>(step.n_process) * m_vocab_size),
                       step,
                       step.n_process);

    // Debug dump outputs
    if (_debug_outputs) {
      if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
        debugOutputs(step, m_layerNames[LayerType::POOL_OUTPUT]);
        debugOutputs(step, m_layerNames[LayerType::SEQ_OUTPUT]);
      } else {
        debugOutputs(step, m_layerNames[LayerType::OUTPUT]);
      }
    }

    n_processed += static_cast<uint32_t>(step.n_process);
    m_inference_count++;
    syncDrafTargetPrefill(draft, true);
  }

  if (post_update) {
    updateFeatureBuffer(embeddingCount);
  }

  // If only the last output is required, then process this request here
  if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
    if (!output_all) {
      getDequantLogits(std::span{output.data(), output.size()}, step, 1);
    }
  } else {
    getEmbeddings(std::span{output.data(), output.size()}, step);
  }

  // Maintain a history of all processed tokens
  if (token_history_enabled)
    token_history.insert(token_history.end(), tokens.begin(), tokens.end());

  __DEBUG("qnn-htp: run-inference complete : {} usec ", start.elapsed_usec());
  return output_size;
}

size_t QnnNspModel::runInference(const std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embedding,
                                 const uint16_t* featureVector,
                                 const std::vector<int32_t>& selected,
                                 uint32_t start_idx,
                                 bool post_update,
                                 const std::vector<int32_t>& attention_map,
                                 Tensor& output,
                                 bool output_all) {
  GENIE_TRACE();
  qualla::Timer start;

  if ((tokens.size() == 0) && (embedding.size() == 0)) return 0;

  size_t embedBufSize   = m_embeddingBufferSize;
  size_t embeddingCount = embedding.size() / embedBufSize;
  // Disable token_history (required for AR-c models) if embedding input is processed
  if (embeddingCount > 0) {
    token_history_enabled = false;
  }

  bool draft = false;
  if (featureVector != nullptr) {
    draft = true;
  }

  // Construct an attention mask processor to simplify handling of complex masks
  const size_t n_inputs = tokens.size() + embeddingCount;
  AttentionMask attention_mask(attention_map,
                               static_cast<size_t>(m_kvmanager->n_past()),
                               static_cast<size_t>(m_kvmanager->n_valid_kv()),
                               n_inputs,
                               _offset_to_apply_kv_prefix,
                               _size_to_skip_kv_prefix);

  if (attention_map.size() > n_inputs && isLongContextEnabled()) {
    State::fatal("LongContext has not been enabled for this dialog");
    return false;
  }

  // Create a strategy to run the inference
  if (!m_kvmanager->prepareInferenceStrategy(static_cast<int32_t>(n_inputs))) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  size_t output_size = output_all ? n_inputs : 1;  // actual number of logits
  output.setSize(0);

  // Loop over each planned iteration in the inference strategy
  InferenceStep step;
  uint32_t n_processed = 0;  // Number of total tokens processed so far

  bool requireLogitsCopy = false;
  if (m_kvmanager->getStrategySize() > 1 && output_all && draft == false) {
    // If we need to return logits from multiple inferences to the caller,
    // then enable logit copying to accumulate logits in the output Tensor.
    requireLogitsCopy = true;
  }

  while (m_kvmanager->nextInferenceStep(step)) {
    if (m_pause && n_processed != 0 && n_processed != 1) {
      m_pause = false;
      return n_processed;
    }

    __DEBUG("Inference step: {}", step.str());
    syncDrafTargetPrefill(draft, false);
    if (!setupInput(step,
                    n_processed,
                    tokens,
                    embedding,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_mask))
      return false;

    for (auto& nsp_graph : m_nsp_graphs) {
      const int graph_idx = nsp_graph.idx();

      if (!nsp_graph.variants.contains({step.variant, step.ctx_size})) {
        continue;
      }

      if (nsp_graph.m_graphType == GraphType::LMHEAD && (!output_all) &&
          !m_kvmanager->isFinalInferenceStep()) {  // Must meet all the criteria to skip LMHEAD
        continue;
      }

      if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }

      if (!nsp_graph.execute(
              step.variant, step.ctx_size, m_inference_count, graph_switching, lazy_lora)) {
        fatal(fmt::format("Failed to execute graph {}", graph_idx));
        return false;
      }

      if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }
    }

    if (output_all) getLogits(output, step, step.n_process, requireLogitsCopy);

    // Debug dump outputs
    if (_debug_outputs) {
      debugOutputs(step, m_layerNames[LayerType::OUTPUT]);
      debugOutputs(step, draftFeatureName);
    }

    n_processed += static_cast<uint32_t>(step.n_process);
    m_inference_count++;
    syncDrafTargetPrefill(draft, true);
  }
  if (post_update) {
    updateFeatureBuffer(embeddingCount);
  }
  // If only the last output is required, then process this request here
  if (!output_all) {
    getLogits(output, step, 1);
  }

  // Maintain a history of all processed tokens
  if (token_history_enabled)
    token_history.insert(token_history.end(), tokens.begin(), tokens.end());

  __DEBUG("qnn-htp: run-inference complete : {} usec ", start.elapsed_usec());
  return output_size;
}

void QnnNspModel::updateFeatureBuffer(uint32_t embeddingCount) {
  size_t feature_len = m_embedding_length * sizeof(uint16_t);
  if (eagle_extra_feature == nullptr) {
    eagle_extra_feature = reinterpret_cast<uint16_t*>(malloc(feature_len));
    std::memset(eagle_extra_feature, 0, feature_len);
  }
  void* eagle_feature = nullptr;
  getIOBufferByName(draftFeatureName, eagle_feature, false);
  const uint16_t* feature_data = nullptr;
  uint32_t feature_offset      = embeddingCount - 1;
  feature_data =
      reinterpret_cast<uint16_t*>(eagle_feature) + feature_offset * feature_len / sizeof(uint16_t);
  std::memcpy(eagle_extra_feature, feature_data, feature_len);
}

// Dumps out the specified tensor to _debug_path numbered according to m_inference_count
bool QnnNspModel::debugOutputs(const InferenceStep& step, const std::string& tensor_name) {
  GENIE_TRACE();
  if (!m_nsp_graphs.back().variants.contains({step.variant, step.ctx_size})) {
    __DEBUG("No outputs found for AR-{} CL-{}", step.variant, step.ctx_size);
    return true;
  }
  GraphVariant* graph_variant = m_nsp_graphs.back()(step.variant, step.ctx_size);
  QnnUtils::Tensor* tensor    = graph_variant->getOutput(tensor_name);
  if (tensor == nullptr) {
    __DEBUG("qnn-htp: Couldn't find tensor {} in graph {}", tensor_name, graph_variant->graph_name);
    return false;
  }

  // For ENCODER models, dump the complete buffer. For LLM models, dump the generated logits
  uint32_t output_bitwidth = tensor->dtype.bw();  // Detect 8-bit vs 16-bit logits
  size_t output_size       = (m_modelArchitectureType == ModelArchitectureType::ENCODER)
                                 ? static_cast<size_t>(step.ctx_size) * output_bitwidth * m_embd_size
                                 : static_cast<size_t>(step.n_process) * output_bitwidth * m_vocab_size;
  std::string fname = fmt::format("{}/{}/{:03d}", _debug_path, tensor_name, m_inference_count);
  if (!QnnUtils::writeRawData(getBuffer(tensor), output_size, fname)) {
    __DEBUG("qnn-htp: Failed to save {}. Error when writing to {}", tensor_name, fname);
    return false;
  }

  return true;
}

bool QnnNspModel::quantizeInput(float* in, size_t tensorOffset, size_t length) {
  if (t_input_ids == nullptr) {
    __ERROR("Input Tensor {} not found during execute", m_layerNames[LayerType::INPUT]);
    return false;
  }

  const auto scale  = t_input_ids->quantParam[0].scale;
  const auto offset = t_input_ids->quantParam[0].offset;
  switch (t_input_ids->dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      QnnUtils::quantizeTensorPtr(in,
                                  reinterpret_cast<uint8_t*>(getBuffer(t_input_ids)) + tensorOffset,
                                  offset,
                                  scale,
                                  length);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      QnnUtils::quantizeTensorPtr(
          in,
          reinterpret_cast<uint16_t*>(getBuffer(t_input_ids)) + tensorOffset,
          offset,
          scale,
          length);
      break;
    default:
      __ERROR("Unsupported alpha tensor dtype {}", t_input_ids->dtype.str());
      return false;
  }

  return true;
}

size_t QnnNspModel::getEmbeddingBufferSize() { return m_embeddingBufferSize; }

void QnnNspModel::getTensorParam(
    LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitwidth) {
  if (layerType == LayerType::INPUT) {
    dataType = t_input_ids->dtype.str();
    scale    = t_input_ids->quantParam[0].scale;
    offset   = t_input_ids->quantParam[0].offset;
    bitwidth = static_cast<size_t>(t_input_ids->dtype.bw());
  }
}

bool QnnNspModel::cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) {
  m_eosEmbedding = eosEmbedding;
  return true;
}

bool QnnNspModel::setKVCacheNPast(size_t n_past, const std::vector<bool>& selected) {
  GENIE_TRACE();
  if (!m_kvmanager->dispatchUpdate(n_past, selected)) {
    __ERROR("qnn-htp: KV$ update failed. {}", m_kvmanager->error());
    State::error(m_kvmanager->error());
    return false;
  }

  // Manage the token history based on the KV$ accepted by the user
  // If no selection mask is passed, we can simply resize the history to n_past
  // If a selection mask is passed, we must selectively filter out rejected KV$
  if (token_history_enabled) {  // Token history must be disabled on embedding input or
                                // longcontext
    if (selected.empty())
      token_history.resize(n_past);
    else {
      auto it = token_history.begin() + static_cast<long>(token_history.size() - selected.size());
      for (const bool& isSelected : selected) {
        it = (isSelected) ? it + 1 : token_history.erase(it);  // Erase if not selected, else no-op
      }
    }
  }
  return true;
}

size_t QnnNspModel::getDequantLogits(std::span<float> buffer, InferenceStep& step, int32_t count) {
  GENIE_TRACE();
  qualla::Timer start;

  QnnUtils::Tensor* const spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)->getOutput(m_layerNames[LayerType::OUTPUT]);
  if (spec == nullptr) {
    State::error("Failed to get output layer tensor spec");
    return 0;
  }

  auto [scale, offset] = spec->quantParam[0];  // Quantization parameters
  QnnUtils::DataType dtype(spec->tensor);      // Datatype of the generated output
  uint32_t bitwidth = spec->dtype.bw();        // Number of bytes per output element
  auto logit_buffer =
      reinterpret_cast<uint8_t*>(getBuffer(spec));  // Pointer to the actual output data

  if (spec->dims.getNumElements() == m_vocab_size && count > 1) {
    State::error("Requested all logits, but graph only produces one logit");
    return 0;
  }

  // Offset to the appropriate location in the output buffer. Note this assumes right-padded input
  logit_buffer += static_cast<uint32_t>(step.n_process - count) * bitwidth * m_vocab_size;

  const size_t size = m_vocab_size * static_cast<size_t>(count);
  __TRACE("qnn-htp: getDequantLogits Returning {}*{} from [{}]", count, m_vocab_size, step.str());

  switch (dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      deQuantizeOutputs(reinterpret_cast<uint8_t*>(logit_buffer), buffer, scale, offset, size);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      deQuantizeOutputs(reinterpret_cast<uint16_t*>(logit_buffer), buffer, scale, offset, size);
      break;
    case QNN_DATATYPE_FLOAT_16:
      castOutputs(reinterpret_cast<uint16_t*>(logit_buffer), buffer, size, bitwidth);
      break;
    case QNN_DATATYPE_FLOAT_32:
      castOutputs(reinterpret_cast<float*>(logit_buffer), buffer, size, bitwidth);
      break;
    default:
      State::error(fmt::format("Unsupported logits dtype {}", dtype.str()));
      return 0;
  }

  __DEBUG("qnn-htp: getDequantLogits complete. Returning {} outputs in {} usec",
          count,
          start.elapsed_usec());
  return size;
}

size_t QnnNspModel::getLogits(Tensor& logits,
                              InferenceStep& step,
                              int32_t count,
                              bool requireLogitsCopy) {
  qualla::Timer start;

  QnnUtils::Tensor* const spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)->getOutput(m_layerNames[LayerType::OUTPUT]);
  if (spec == nullptr) {
    State::error("Failed to get output layer tensor spec");
    return 0;
  }

  auto [scale, offset] = spec->quantParam[0];  // Quantization parameters
  QnnUtils::DataType dtype(spec->tensor);      // Datatype of the generated output
  uint32_t bitwidth = spec->dtype.bw();        // Number of bytes per output element
  auto logit_buffer =
      reinterpret_cast<uint8_t*>(getBuffer(spec));  // Pointer to the actual output data

  if (spec->dims.getNumElements() == m_vocab_size && count > 1) {
    State::error("Requested all logits, but graph only produces one logit");
    return 0;
  }

  // Offset to the appropriate location in the output buffer. Note this assumes right-padded input
  logit_buffer += static_cast<uint32_t>(step.n_process - count) * m_vocab_size * bitwidth;

  const size_t size = m_vocab_size * static_cast<size_t>(count);
  __TRACE("qnn-htp: getLogits Returning {}*{} from [{}]", count, m_vocab_size, step.str());

  switch (dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        uint8_t* logit_buffer_u8 = reinterpret_cast<uint8_t*>(logit_buffer);
        for (uint32_t i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = static_cast<float>(scale) *
              (static_cast<float>(logit_buffer_u8[i]) + static_cast<float>(offset));
        }
        logits.setQuantizationParams(1, 0);
        logits.setData(static_cast<void*>(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      } else {
        logits.setQuantizationParams(scale, offset);
        logits.setData(static_cast<void*>(logit_buffer));
        logits.setSize(size);
        logits.setDataType(TENSOR_DATATYPE_UFIXED_POINT_8);
      }
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        uint16_t* logit_buffer_u16 = reinterpret_cast<uint16_t*>(logit_buffer);
        for (uint32_t i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = static_cast<float>(scale) *
              (static_cast<float>(logit_buffer_u16[i]) + static_cast<float>(offset));
        }
        logits.setQuantizationParams(1, 0);
        logits.setData(reinterpret_cast<void*>(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      } else {
        logits.setQuantizationParams(scale, offset);
        logits.setData(reinterpret_cast<void*>(logit_buffer));
        logits.setSize(size);
        logits.setDataType(TENSOR_DATATYPE_UFIXED_POINT_16);
      }
      break;
    }
    case QNN_DATATYPE_FLOAT_16: {
      // Downstream tasks (like sampling) can't handle float16 yet. Always convert to float32.
      logits.logits.reserve(logits.getSize() + size);
      uint16_t* logit_buffer_fp16 = reinterpret_cast<uint16_t*>(logit_buffer);
      for (uint32_t i = 0; i < size; i++) {
        logits.logits[logits.getSize() + i] = fp16_ieee_to_fp32_value(logit_buffer_fp16[i]);
      }
      logits.setQuantizationParams(1, 0);
      logits.setData(reinterpret_cast<void*>(logits.logits.data()));
      logits.setSize(logits.getSize() + size);
      logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        float* logit_buffer_fp32 = reinterpret_cast<float*>(logit_buffer);
        for (uint32_t i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = logit_buffer_fp32[i];
        }
        logits.setData(reinterpret_cast<void*>(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
      } else {
        logits.setData(reinterpret_cast<void*>(logit_buffer));
        logits.setSize(size);
      }
      logits.setQuantizationParams(1, 0);
      logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      break;
    }
    default: {
      State::error(fmt::format("Unsupported logits dtype {}", dtype.str()));
      return 0;
    }
  }

  __DEBUG(
      "qnn-htp: getLogits complete. Returning {} outputs in {} usec", count, start.elapsed_usec());
  return size;
}

bool QnnNspModel::calculate_rope_embeddings(void) {
  if (m_positional_encoding.type != PositionalEncoding::ROPE) {
    return true;
  }
  if (m_lazyInitialization || m_ropeInitialized) return true;
  const size_t nmemb    = m_ctx_size * m_pos_dim;
  const uint32_t pos_bw = d_pos.bw();

  const double theta                    = m_positional_encoding.rope_params.theta;
  const RopeScalingParams& rope_scaling = m_positional_encoding.rope_params.rope_scaling;

  rope_sin = malloc(nmemb * pos_bw);
  rope_cos = malloc(nmemb * pos_bw);

  auto [q_scale, q_offset] = t_position_ids_cos->quantParam[0];
  if (d_pos == QNN_DATATYPE_FLOAT_16 || d_pos == QNN_DATATYPE_FLOAT_32) {
    // If floating point, don't quantize!
    q_scale  = 1.0;
    q_offset = 0;
  }

  // Calculate inv_freq array
  std::vector<double> inv_freq(m_pos_dim);
  const double exponent = 1.0 / static_cast<double>(m_pos_dim);
  for (uint32_t j = 0; j < m_pos_dim; j++) {
    inv_freq[j] = 1.0 / pow(theta, j * exponent);
  }
  double attention_factor = 1.0;
  if (rope_scaling.rope_type == RopeScalingParams::ROPE_LLAMA3) {
    // Implemented from HuggingFace
    // https://github.com/huggingface/transformers/blob/47c29ccfaf56947d845971a439cbe75a764b63d7/src/transformers/modeling_rope_utils.py#L298
    const double& factor           = rope_scaling.llama3_params.factor;
    const double& low_freq_factor  = rope_scaling.llama3_params.low_freq_factor;
    const double& high_freq_factor = rope_scaling.llama3_params.high_freq_factor;
    const int& old_context_len     = rope_scaling.llama3_params.original_max_position_embeddings;

    const double low_freq_wavelen  = old_context_len / low_freq_factor;
    const double high_freq_wavelen = old_context_len / high_freq_factor;

    for (uint32_t j = 0; j < m_pos_dim; j++) {
      const double wavelen = 2 * M_PI / inv_freq[j];
      if (wavelen < high_freq_wavelen)  // wavelen < high_freq_wavelen: do nothing
        continue;
      else if (wavelen > low_freq_wavelen)  // wavelen > low_freq_wavelen: divide by factor
        inv_freq[j] = 1.0 / static_cast<double>(factor * pow(theta, j * exponent));
      else {  // otherwise: interpolate between the two, using a smooth factor
        assert(low_freq_wavelen != high_freq_wavelen);
        const double smooth = (static_cast<double>(old_context_len) / wavelen - low_freq_factor) /
                              (high_freq_factor - low_freq_factor);
        inv_freq[j] = ((1 - smooth) * inv_freq[j] / factor + smooth * inv_freq[j]);
      }
    }
  } else if (rope_scaling.rope_type == RopeScalingParams::ROPE_LONGROPE) {
    // Validate factor >= 1.0, len(long_factor) == rope-dim and len(short_factor) == rope-dim
    const double& factor       = rope_scaling.longrope_params.factor;
    const int& old_context_len = rope_scaling.longrope_params.original_max_position_embeddings;

    const auto& inv_factors = (m_ctx_size > static_cast<size_t>(old_context_len))
                                  ? rope_scaling.longrope_params.long_factor
                                  : rope_scaling.longrope_params.short_factor;

    if (inv_factors.size() != m_pos_dim)
      throw std::runtime_error(
          fmt::format("long-factor (len={}) and short-factor (len={}) must have length rope-dim={}",
                      rope_scaling.longrope_params.long_factor.size(),
                      rope_scaling.longrope_params.short_factor.size(),
                      m_pos_dim));

    for (uint32_t j = 0; j < m_pos_dim; j++) {
      inv_freq[j] = inv_freq[j] / inv_factors[j];
    }

    attention_factor =
        std::sqrt(1.0 + std::log(factor) / std::log(static_cast<double>(old_context_len)));
  }
  for (uint32_t i = 0; i < m_ctx_size; i++) {
    for (uint32_t j = 0; j < m_pos_dim; j++) {
      const double freq = i * inv_freq[j];

      const double sin_val = ((sin(freq) * attention_factor) / q_scale) - q_offset;
      const double cos_val = ((cos(freq) * attention_factor) / q_scale) - q_offset;

      // round() instead of floor() seems to produce an acuracy drop. To debug later
      switch (d_pos) {
        case QNN_DATATYPE_UFIXED_POINT_8:
          (reinterpret_cast<uint8_t*>(rope_sin))[i * m_pos_dim + j] = static_cast<uint8_t>(sin_val);
          (reinterpret_cast<uint8_t*>(rope_cos))[i * m_pos_dim + j] = static_cast<uint8_t>(cos_val);
          break;
        case QNN_DATATYPE_UFIXED_POINT_16:
          (reinterpret_cast<uint16_t*>(rope_sin))[i * m_pos_dim + j] =
              static_cast<uint16_t>(sin_val);
          (reinterpret_cast<uint16_t*>(rope_cos))[i * m_pos_dim + j] =
              static_cast<uint16_t>(cos_val);
          break;
        case QNN_DATATYPE_FLOAT_16:
          (reinterpret_cast<uint16_t*>(rope_sin))[i * m_pos_dim + j] =
              fp16_ieee_from_fp32_value(sin_val);
          (reinterpret_cast<uint16_t*>(rope_cos))[i * m_pos_dim + j] =
              fp16_ieee_from_fp32_value(cos_val);
          break;
        case QNN_DATATYPE_FLOAT_32:
          (reinterpret_cast<float*>(rope_sin))[i * m_pos_dim + j] = static_cast<float>(sin_val);
          (reinterpret_cast<float*>(rope_cos))[i * m_pos_dim + j] = static_cast<float>(cos_val);
          break;
        default:
          __ERROR("Unsupported position ids datatype {}", d_pos.str());
          return false;
      }
    }
  }

  if (_debug_tensors) {
    std::string dtype =
        fmt::format("{}{}", (d_pos == QNN_DATATYPE_FLOAT_16) ? "f" : "u", pos_bw * 8);
    std::string fname_sin = fmt::format("{}/position_ids_sin.{}.dat", _debug_path, dtype);
    std::string fname_cos = fmt::format("{}/position_ids_cos.{}.dat", _debug_path, dtype);
    QnnUtils::writeRawData(rope_sin, nmemb * pos_bw, fname_sin);
    QnnUtils::writeRawData(rope_cos, nmemb * pos_bw, fname_cos);
  }

  m_ropeInitialized = true;
  return true;
}

bool QnnNspModel::load_lmhead_weight_as_input(void) {
  if (!_lmhead_weight_input) return true;
  if (_lmhead_weight_input && lmhead_weight_dir.empty()) {
    __ERROR("NSPModel: LMhead weight file not found");
    return false;
  }
  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      if (tname.compare("weight") == 0) {
        // weight tensor file name should be in same format as tensor name present in graph
        std::string weight_file =
            (model_basedir / fs::path(lmhead_weight_dir) / fs::path(tname + ".raw")).string();

        QnnUtils::Dims dims = tspec.dims;
        size_t numElements  = dims.getNumElements();

        size_t size = sizeof(float);
        std::vector<float> weight_f32;  // Temporary variable to load fp32 values
        weight_f32.resize(numElements);

        FILE* fp = fopen(weight_file.c_str(), "rb");
        if (fp == NULL) {
          __ERROR("NSPModel: Error opening file: {}", weight_file);
          return false;
        }

        size_t count = fread(weight_f32.data(), size, numElements, fp);
        fclose(fp);

        if (count != numElements) {
          __ERROR("NSPModel: Could not load {} - expected file size {}",
                  weight_file,
                  numElements * size);
          return false;
        }

        int8_t* weight_buffer = reinterpret_cast<int8_t*>(getBuffer(tspec));
        // Quantize the values, per width quantization
        QnnUtils::perWidthQuantizeTensorPtr(weight_f32.data(),
                                            weight_buffer,
                                            tspec.quantParam,
                                            dims.height,
                                            dims.width,
                                            dims.channel);
      }
    }
  }
  return true;
}

void QnnNspModel::getInputQuantParam(double& scale, int& offset) {
  auto tmp = t_input_ids->quantParam[0];
  scale    = tmp.scale;
  offset   = tmp.offset;
}

size_t QnnNspModel::loadKVCache(const std::string& load_path, bool /*chooseHigherVariant*/) {
  m_kvmanager->block(Scope::global());
  size_t ret = m_kvmanager->loadKVCache(load_path);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::saveKVCache(const std::string& save_path) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->dumpKVCache(save_path);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::saveKVCacheToBuffer(Buffer* kvBuff) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->dumpKVCache(kvBuff);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::getCacheSpec(CacheFileSpec& spec) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->getCacheSpec(spec);
  return ret;
}

bool QnnNspModel::getKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->getKVHead(spec, layer, head, data, scale);
  return ret;
}

void QnnNspModel::setHigherVariant() {
  auto& [new_variant, _] = nsp_graph_count.rbegin()->first;  // Guarantees largest variant, then ctx
  m_kvmanager->setActiveVariant(new_variant, -1);
}

size_t QnnNspModel::getEmbeddings(std::span<float> embds, InferenceStep& step) {
  qualla::Timer start;

  QnnUtils::Tensor* output_spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)
          ->getOutput(m_pooled_output ? m_layerNames[LayerType::POOL_OUTPUT]
                                      : m_layerNames[LayerType::SEQ_OUTPUT]);

  if (output_spec == nullptr) {
    __ERROR("encountered null buffer");
    throw std::runtime_error("Model is not supporting per token embedding");
  }
  const auto scale  = output_spec->quantParam[0].scale;
  const auto offset = output_spec->quantParam[0].offset;

  auto output_datatype   = QnnUtils::DataType(output_spec->tensor);
  uint32_t output_bw     = output_spec->dtype.bw();
  uint8_t* output_buffer = reinterpret_cast<uint8_t*>(getBuffer(output_spec));

  const int return_size = m_pooled_output ? 1 : step.n_process;
  if (!m_pooled_output) {
    // If multiple tokens embedding are returned, offset to the correct location in the buffer
    if (step.variant == step.ctx_size) {
      // This was left-padded, tokens embedding are at [n_tokens - n_processed, n_tokens]
      output_buffer += static_cast<uint32_t>(step.variant - return_size) * m_embd_size * output_bw;
    } else {
      // This was right-padded, tokens embedding are at indexes [0, n_processed]
      output_buffer += static_cast<uint32_t>(step.n_process - 1) * m_embd_size * output_bw;
    }
  }

  const size_t output_len = static_cast<size_t>(return_size) * m_embd_size;
  __TRACE("qnn-htp: get-embds for {} tokens. scale = {}, offset = {}, Returning {}",
          step.n_process,
          scale,
          offset,
          output_len);

  switch (output_datatype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      deQuantizeOutputs(
          reinterpret_cast<uint8_t*>(output_buffer), embds, scale, offset, output_len);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      deQuantizeOutputs(
          reinterpret_cast<uint16_t*>(output_buffer), embds, scale, offset, output_len);
      break;
    case QNN_DATATYPE_FLOAT_16:
      castOutputs(reinterpret_cast<uint16_t*>(output_buffer), embds, output_len, output_bw);
      break;
    case QNN_DATATYPE_FLOAT_32:
      castOutputs(reinterpret_cast<float*>(output_buffer), embds, output_len, output_bw);
      break;
    default:
      __ERROR("Unsupported output datatype");
  }

  __DEBUG("qnn-htp: getEmbeddings complete : {} usec (return_size={})",
          start.elapsed_usec(),
          output_len);
  return output_len;
}

size_t QnnNspModel::getIOBufferByName(std::string tensor_name, void*& buffer, bool isPrompt) {
  int32_t token =
      isPrompt ? nsp_graph_count.rbegin()->first.first : nsp_graph_count.begin()->first.first;
  int32_t ctxt =
      isPrompt ? nsp_graph_count.rbegin()->first.second : nsp_graph_count.begin()->first.second;
  __DEBUG("getIOBufferByName isPrompt {} token {} ctxt {}", isPrompt, token, ctxt);

  for (QnnNspGraph& graph : m_nsp_graphs) {
    GraphVariant* variant = graph(token, ctxt);
    if (variant->getOutput(tensor_name) != nullptr) {
      buffer             = getBuffer(variant->getOutput(tensor_name));
      size_t buffer_size = getBufferSize(variant->getOutput(tensor_name));
      __DEBUG("qnn-htp: getIOBufferByNam output tensor_name {} address {} buffer_size {}",
              tensor_name,
              reinterpret_cast<uintptr_t>(buffer),
              buffer_size);

      break;
    }
    if (variant->getInput(tensor_name) != nullptr) {
      buffer             = getBuffer(variant->getInput(tensor_name));
      size_t buffer_size = getBufferSize(variant->getInput(tensor_name));
      __DEBUG("qnn-htp: getIOBufferByNam input tensor_name {} address {} buffer_size {}",
              tensor_name,
              reinterpret_cast<uintptr_t>(buffer),
              buffer_size);
      break;
    }
  }

  return static_cast<size_t>(token);
}

bool QnnNspModel::finalizeState(std::shared_ptr<EngineState>& engineState) {
  IOEVENT event = engineState->isInitialize() ? engineState->getIOBuffer()->m_event
                                              : IOEVENT::ALLOCATE_REGISTER_EVENT;

  __DEBUG("qnn-htp: Event triggered {}", ioEventMap[event]);
  if (event == IOEVENT::NO_EVENT) {
    return true;
  }

  if (m_kvmanager) {
    m_kvmanager->deRegisterAll();
  }

  if (true != QnnNspBaseModel::finalizeState(engineState)) {
    return false;
  }

  m_lazyInitialization = false;

  if (!initializeIOTensors()) {
    __ERROR("Error in re-initializing the Tensors");
    return false;
  }

  if (event == IOEVENT::ALLOCATE_REGISTER_EVENT) {
    // reinitialize the KV manager or initialize
    if (true != initializeKVManager()) {
      __ERROR("Error in allocating the KV manager memory");
      return false;
    }
    if (true != initializeTensorPointers()) {
      __ERROR("Error in initializing Tensor pointers");
      return false;
    }
    if (true != calculate_rope_embeddings()) {
      __ERROR("Error in creating Rope Data");
      return false;
    }
    engineState->initialize(std::dynamic_pointer_cast<IOBuffer>(m_kvmanager));

  } else if (event == IOEVENT::REGISTER_EVENT) {
    m_kvmanager = std::dynamic_pointer_cast<KVManager>(engineState->getIOBuffer());
    // might need to update some static fields.
    if (true != initializeTensorPointers()) {
      __ERROR("Error in initializing Tensor pointers");
      return false;
    }
    if (true != calculate_rope_embeddings()) {
      __ERROR("Error in creating Rope Data");
      return false;
    }
  }
  // always change event to NOEVENT after all processing is done
  if (true != engineState->changeIOEvent(IOEVENT::NO_EVENT)) {
    __ERROR("Error: Failed to set IO Event for engine states");
    return false;
  }

  m_lazyInitialization = true;

  return true;
}

}  // namespace qualla
