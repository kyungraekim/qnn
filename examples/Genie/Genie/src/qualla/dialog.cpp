//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <fstream>

#include "Trace.hpp"
#include "TraceLogger.hpp"
#include "qualla/detail/timer.hpp"

// Dialogs
#include "dialogs/basic.hpp"
#include "dialogs/eaglet.hpp"
#include "dialogs/kv-share.hpp"
#include "dialogs/lhd-dec.hpp"
#include "dialogs/multistream.hpp"
#include "dialogs/spec-dec.hpp"
#include "dialogs/ssd-q1.hpp"
#include "qualla/dialog.hpp"

#define __ERROR(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __KVTRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace fs = std::filesystem;

namespace qualla {

Dialog::Dialog(std::shared_ptr<Env> env, const std::string& name, const qualla::json& json)
    : State(env->getTraceLogger()), _env(env) {
  GENIE_TRACE();
  Timer start;

  __DEBUG("dialog-new: {} config {}", name, json.dump());

  using qc = qualla::Config;

  // Create Gpiomarker and reset the gpio status to low
  const qualla::json& gpio_conf = qc::optional<qualla::json>(json, "gpio", {});
  _gpio_marker                  = GpioMarker::create(gpio_conf);

  _gpio_marker->set();

  // Create the context first
  _ctx = Context::create(_env, name, qc::mandatory<qualla::json>(json, "context"));

  // Parse prompt config
  const qualla::json& pmt_conf = qc::optional<qualla::json>(json, "prompt", {});
  _prompt_type                 = qc::optional<std::string>(pmt_conf, "type", "llama2");
  _sys_tags   = qc::optional<std::vector<std::string>>(pmt_conf, "sys-tags", {"", ""});
  _inst_tags  = qc::optional<std::vector<std::string>>(pmt_conf, "inst-tags", {"", ""});
  _role_tags  = qc::optional<std::vector<std::string>>(pmt_conf, "role-tags", {"", ""});
  _sys_prompt = qc::optional<std::string>(pmt_conf, "sys-prompt", "");

  const std::vector<std::string>& stop_sequence =
      qc::optional<std::vector<std::string>>(pmt_conf, "stop-sequence", {});
  _stop_sequence = SequenceMatchTrie(stop_sequence);

  // Create Tokenizer
  // TODO: auto-detect / validate n_vocab with tokenizer vocab
  fs::path tok_path = _env->path().models / qc::mandatory<std::string>(json, "tokenizer");
  _tokenizer        = Tokenizer::create(*_ctx, tok_path);

  // Create Sampler(s)
  auto add_sampler = [&](const qualla::json& j) {
    std::string role = qc::optional<std::string>(j, "role", "primary");
    _sampler[role]   = Sampler::create(*_ctx, j);
  };

  const qualla::json& sam_conf = qc::mandatory<qualla::json>(json, "sampler");
  if (sam_conf.is_array()) {
    for (auto sc : sam_conf) {
      add_sampler(sc);
    }
  } else
    add_sampler(sam_conf);

  // Create Engine(s)
  auto add_engine = [&](const qualla::json& j) {
    std::string role = qc::optional<std::string>(j, "role", "primary");

    _engine[role] = Engine::create(*_ctx, j);
    using FF      = Engine::Feature::Flags;

    if (!_engine[role]->supports(FF::OUTPUT_LOGITS))
      throw std::runtime_error("the engine must output Logits");
    // Mark it bound for the current dialog.
    _engine[role]->bound();
  };

  const qualla::json& eng_conf = qc::optional<qualla::json>(json, "engine", {});

  if (!eng_conf.empty()) {
    if (eng_conf.is_array()) {
      for (auto ec : eng_conf) {
        add_engine(ec);
      }
    } else {
      add_engine(eng_conf);
    }
  }

  // Encoder translation for LUT + decoder flow
  qualla::json encoder_conf = qc::optional<qualla::json>(json, "encoder", {});
  if (encoder_conf.contains("type")) {
    if (encoder_conf["type"] == "lut") {
      _encoder = Encoder::create(_env, name, encoder_conf);
      // Will be replaced by tensor info from LUT post tensor changes merged
      lutDataType = encoder_conf["context"]["embedding-datatype"];
      if (encoder_conf["context"].contains("quant-param")) {
        lutScale  = encoder_conf["context"]["quant-param"]["scale"];
        lutOffset = encoder_conf["context"]["quant-param"]["offset"];
      }
      calculateRequantEncodings();
      m_t2eCallbacks["QNN_DATATYPE_FLOAT_32"]["QNN_DATATYPE_FLOAT_32"] =
          &Dialog::tokenToEmbedCallback;
      m_t2eCallbacks["QNN_DATATYPE_SFIXED_POINT_8"]["QNN_DATATYPE_SFIXED_POINT_8"] =
          &Dialog::tokenToEmbedRequantCallback<int8_t, int8_t>;
      m_t2eCallbacks["QNN_DATATYPE_SFIXED_POINT_8"]["QNN_DATATYPE_SFIXED_POINT_16"] =
          &Dialog::tokenToEmbedRequantCallback<int8_t, int16_t>;
      m_t2eCallbacks["QNN_DATATYPE_UFIXED_POINT_8"]["QNN_DATATYPE_UFIXED_POINT_8"] =
          &Dialog::tokenToEmbedRequantCallback<uint8_t, uint8_t>;
      m_t2eCallbacks["QNN_DATATYPE_UFIXED_POINT_8"]["QNN_DATATYPE_UFIXED_POINT_16"] =
          &Dialog::tokenToEmbedRequantCallback<uint8_t, uint16_t>;
      m_t2eCallbacks["QNN_DATATYPE_SFIXED_POINT_16"]["QNN_DATATYPE_SFIXED_POINT_8"] =
          &Dialog::tokenToEmbedRequantCallback<int16_t, int8_t>;
      m_t2eCallbacks["QNN_DATATYPE_SFIXED_POINT_16"]["QNN_DATATYPE_SFIXED_POINT_16"] =
          &Dialog::tokenToEmbedRequantCallback<int16_t, int16_t>;
      m_t2eCallbacks["QNN_DATATYPE_UFIXED_POINT_16"]["QNN_DATATYPE_UFIXED_POINT_8"] =
          &Dialog::tokenToEmbedRequantCallback<uint16_t, uint8_t>;
      m_t2eCallbacks["QNN_DATATYPE_UFIXED_POINT_16"]["QNN_DATATYPE_UFIXED_POINT_16"] =
          &Dialog::tokenToEmbedRequantCallback<uint16_t, uint16_t>;
    }
  }
  if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_8" ||
      lutDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
    lutByteWidth = 1;
  } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_16" ||
             lutDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
    lutByteWidth = 2;
  } else if (lutDataType == "QNN_DATATYPE_FLOAT_32") {
    lutByteWidth = 4;
  }
  for (auto& engine : _engine) {
    engine.second->getPerfProfile(m_defaultPerfProfile);
  }

  m_perfProfile = m_defaultPerfProfile;

  // create lora Config
  auto add_loraConfig = [&](auto& cur) {
    auto role = qc::optional<std::string>(cur, "role", "primary");
    Config config(cur, "dialog-loraConfig");
    m_loraConfig[role] = std::make_shared<LoraConfig>(config, _env);
  };
  auto loraConfig = qc::optional<qualla::json>(json, "loraConfig", {});
  if (!loraConfig.empty()) {
    if (loraConfig.is_array()) {
      for (auto& cur : loraConfig) add_loraConfig(cur);
    } else
      add_loraConfig(loraConfig);
  }

  completeInit();

  _kpis.init.update(start.elapsed_usec());
}

Dialog::~Dialog() {}

static bool __no_response_token(const int32_t*, const uint32_t, Sentence::Code) { return false; }

static bool __no_response(const std::string&, Sentence::Code) { return false; }

void Dialog::completeInit() {
  // Store input type (token, embedding, etc) from the engine.
  // This assumes multi-engine usecases use matching input types.
  if (!_engine.empty()) m_inputType = _engine.begin()->second->getInputType();

  if (m_traceLogger) {
    setTraceLogger(m_traceLogger);
  }
}
void Dialog::addSupplementInitTime(uint64_t extraInitTime) { _kpis.init.update(extraInitTime); }

void Dialog::getTopK(Tensor& logits,
                     std::vector<std::vector<int32_t>>& tokens,
                     size_t topK,
                     float pThreshold,
                     Dialog::Callback callback) {
  switch (logits.getDataType()) {
    case TENSOR_DATATYPE_UFIXED_POINT_8: {
      return runTopK<uint8_t>(logits, tokens, topK, pThreshold, callback);
    }
    case TENSOR_DATATYPE_UFIXED_POINT_16: {
      return runTopK<uint16_t>(logits, tokens, topK, pThreshold, callback);
    }
    case TENSOR_DATATYPE_FLOAT_POINT_16: {
      return runTopK<uint16_t>(logits, tokens, topK, pThreshold, callback);
    }
    case TENSOR_DATATYPE_FLOAT_32: {
      return runTopK<float>(logits, tokens, topK, pThreshold, callback);
    }
    default: {
      std::cerr << "Unsupported logits datatype" << std::endl;
    }
  }
}

template <typename T>
void Dialog::runTopK(Tensor& logits,
                     std::vector<std::vector<int32_t>>& tokens,
                     size_t topK,
                     float pThreshold,
                     Dialog::Callback callback) {
  auto& sampler = *_sampler["primary"];

  IndexedQuantLogits<T> indexed_logits(logits, sampler.rng(), sampler.getPenalty());
  indexed_logits.penalizeLogits();
  indexed_logits.softmax();
  indexed_logits.topK(topK);

  for (size_t i = 0; i < topK; i++) {
    _last_tok = indexed_logits.indices[i];

    // Only sample tokens above some probability threshold
    // TODO: Modify sampling algorithm as necessary
    if (indexed_logits.probs[i] < pThreshold) {
      break;
    } else if (_ctx->is_eos(_last_tok)) {
      callback("", Sentence::CONTINUE);
    } else {
      tokens.push_back({_last_tok});
      sampler.updateSampledTokenHistory(_last_tok, i);
    }
  }
}

void Dialog::calculateRequantEncodings() {
  if (_engine.empty()) return;
  _engine.begin()->second->getTensorParam(
      LayerType::INPUT, inputDataType, inputScale, inputOffset, inputBitWidth);
  requantScale  = lutScale / inputScale;
  requantOffset = requantScale * lutOffset - inputOffset;
}

void Dialog::inputTensorQuantParam(std::string& dataType,
                                   double& scale,
                                   int32_t& offset,
                                   size_t& byteWidth) {
  if (lutDataType == "QNN_DATATYPE_FLOAT_32") {
    dataType  = "QNN_DATATYPE_FLOAT_32";
    scale     = 1.0;
    offset    = 0;
    byteWidth = 4;
  } else {
    dataType  = inputDataType;
    scale     = inputScale;
    offset    = inputOffset;
    byteWidth = inputBitWidth;
  }
}

void Dialog::pauseQuery() {
  m_pause = true;
  for (auto& engine : _engine) {
    engine.second->pauseQuery();
  }
}

void Dialog::setPerformancePolicy(qualla::PerformanceProfile policy) {
  m_perfProfile = policy;
  for (auto& engine : _engine) {
    engine.second->setPerfProfile(policy);
  }
}

qualla::PerformanceProfile& Dialog::getPerformancePolicy() { return m_perfProfile; }

void Dialog::requantEmbedding(void* from, void* to, size_t length) {
  for (size_t i = 0; i < length; i++) {
    if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
        inputDataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t*>(to)[i] =
          static_cast<int8_t>(requantScale * static_cast<int8_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
               inputDataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t*>(to)[i] =
          static_cast<int16_t>(requantScale * static_cast<int8_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               inputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t*>(to)[i] =
          static_cast<uint8_t>(requantScale * static_cast<uint8_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               inputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t*>(to)[i] =
          static_cast<uint16_t>(requantScale * static_cast<uint8_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               inputDataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t*>(to)[i] =
          static_cast<int8_t>(requantScale * static_cast<int16_t*>(from)[i] + requantOffset);
    } else if (lutDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               inputDataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t*>(to)[i] =
          static_cast<int16_t>(requantScale * static_cast<int16_t*>(from)[i] + requantOffset);
    }
    if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
        inputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t*>(to)[i] =
          static_cast<uint8_t>(requantScale * static_cast<uint16_t*>(from)[i] + requantOffset);
    }
    if (lutDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
        inputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t*>(to)[i] =
          static_cast<uint16_t>(requantScale * static_cast<uint16_t*>(from)[i] + requantOffset);
    }
  }
}

void Dialog::tokenToEmbedCallback(int32_t token, void* embedding, size_t embeddingSize) {
  uint32_t lutIndex = static_cast<uint32_t>(token) * embeddingSize;
  if ((lutIndex + embeddingSize) <= _encoder->getEmbeddingLutSize()) {
    int8_t* embeddingSrc = static_cast<int8_t*>(_encoder->getEmbeddingLut()) + lutIndex;
    int8_t* embeddingDst = static_cast<int8_t*>(embedding);
    std::copy(embeddingSrc, embeddingSrc + embeddingSize, embeddingDst);
  } else {
    std::cerr << "Error: T2E conversion overflow." << std::endl;
  }
}

template <class F, class T>
void Dialog::tokenToEmbedRequantCallback(int32_t token, void* embedding, size_t embeddingSize) {
  size_t numElements = embeddingSize / sizeof(T);
  uint32_t lutIndex  = static_cast<uint32_t>(token) * numElements;
  if ((lutIndex + numElements) * sizeof(F) <= _encoder->getEmbeddingLutSize()) {
    F* embeddingSrc = static_cast<F*>(_encoder->getEmbeddingLut()) + lutIndex;
    T* embeddingDst = static_cast<T*>(embedding);
    if (lutDataType == inputDataType && requantScale == 1 && requantOffset == 0) {
      // Skip requant if quant parameters are not different
      std::copy(embeddingSrc, embeddingSrc + numElements, embeddingDst);
    } else {
      requantEmbedding(embeddingSrc, embeddingDst, numElements);
    }
  } else {
    std::cerr << "Error: T2E conversion overflow." << std::endl;
  }
}

bool Dialog::query(const std::string& str, Sentence::Code scode, Dialog::Callback callback) {
  // always reset before start.
  m_rewindAtBoundary = false;

  // LUT + E2T invocation
  if (_encoder && _encoder->type() == "lut") {
    std::vector<uint8_t> encoderOutput;
    std::vector<uint8_t> decoderInput;
    std::vector<int32_t> tokenizedInput;
    bool status = _encoder->encode(str, encoderOutput, tokenizedInput);
    addPromptTokenHistory(tokenizedInput);
    Dialog::T2ECallback t2eCallback =
        m_t2eCallbacks["QNN_DATATYPE_FLOAT_32"]["QNN_DATATYPE_FLOAT_32"];  // default callback
    if (lutDataType != "QNN_DATATYPE_FLOAT_32") {
      t2eCallback = m_t2eCallbacks[lutDataType][inputDataType];
      if (lutDataType == inputDataType && requantScale == 1 && requantOffset == 0) {
        // Skip requantization if the LUT and input encodings are identical.
        decoderInput = std::move(encoderOutput);
      } else {  // requantize data if encoding parameters are different
        size_t numElements = encoderOutput.size() / lutByteWidth;
        decoderInput.resize(numElements * inputBitWidth);
        requantEmbedding(encoderOutput.data(), decoderInput.data(), numElements);
      }
    } else {
      decoderInput = std::move(encoderOutput);
    }
    if (status == false) return status;
    return query(decoderInput, scode, t2eCallback, callback);
  }
  std::vector<int32_t> p_vec;  // prompt tokens
  std::string p_str;           // prompt string

  p_vec.reserve(1024);

  _tokenizer->cleanUp();  // clean dangling history of the previous queries.

  if (scode == Sentence::REWIND) {
    // this is case when query does not match but the query will continue
    // with existing prompt as fresh
    // case where the last query had sys tag added now for
    // current also need to add sys prompt for matching
    _n_queries = 0;
    _last_tok  = -1;
  }
  if (scode == Sentence::COMPLETE || scode == Sentence::BEGIN || scode == Sentence::REWIND) {
    // Reset prompt/gen counts for new query
    _n_prompt             = 0;
    _n_generated          = 0;
    _n_previous_prompt    = 0;
    _n_previous_generated = 0;

    if (_last_tok >= 0 && !_ctx->is_eos(_last_tok) &&
        !detectedStopSeq)  // avoid putting stop sequence in query
      p_vec.push_back(_last_tok);
    detectedStopSeq = false;

    p_str = _inst_tags[0];

    if (!_n_queries) {
      // First query. Prepend sys-prompt.
      p_str += _sys_tags[0] + _sys_prompt + _sys_tags[1];
    } else {
      // Add EOS explicitly if the last query was aborted prematurely.
      if (_ctx->eos_tok() >= 0) p_vec.push_back(_ctx->eos_tok());
    }

    // Add BOS
    if (_ctx->bos_tok() >= 0) {
      p_vec.push_back(_ctx->bos_tok());
    }
  }

  // FIXME: make this more generic
  if (_prompt_type == "llama3") {
    p_str += _sys_tags[0] + _role_tags[1] + _sys_tags[1] + str + _inst_tags[2];
  } else {
    p_str += str;
  }

  if (scode == Sentence::COMPLETE || scode == Sentence::END || scode == Sentence::REWIND) {
    if (_prompt_type == "llama3") {
      p_str += _sys_tags[0] + _role_tags[2] + _sys_tags[1];
    } else {
      p_str += _inst_tags[1];
    }
  }

  qualla::json j{{"prompt", p_str}};
  __DEBUG("dialog-query: {} {}", _ctx->name(), j.dump());

  _n_queries++;

  if (scode != Sentence::RESUME) {
    if (m_processState != NO_RESUME) {
      return Dialog::abort("Need to resume a paused query. ", callback);
    }
    _tokenizer->encode(p_str, p_vec);
  } else {
    if (!supportsPauseResume()) {
      return Dialog::abort("Pause/Resume is not supported on this dialog. ", callback);
    }
    if (m_processState == NO_RESUME) {
      return Dialog::abort("Cannot resume a query which is not paused. ", callback);
    }
    p_vec = m_unprocessedTokens;
    __DEBUG("Resuming dialog-query with: {}", p_vec);
  }

  __DEBUG("dialog-tokens: {} {}", _ctx->name(), p_vec);
  __DEBUG("dialog-text: \"{}\"", p_str);
  if (scode == Sentence::REWIND) {
    kvRewindPrefixMatch(p_vec);
    __DEBUG("dialog-tokens-after-KV$-rewind: {} {}", _ctx->name(), p_vec);
  }

  if (scode == Sentence::COMPLETE || scode == Sentence::END || scode == Sentence::REWIND ||
      scode == Sentence::RESUME) {
    // Detect stop sequences here
    if (!_stop_sequence.empty()) {
      _stop_sequence.reset();
      Dialog::Callback stopSeqCallback = [this, callback](const std::string& s, Sentence::Code c) {
        return getStopSeqCallback(s, c, callback);
      };
      addPromptTokenHistory(p_vec);
      auto returnVal = process(p_vec, stopSeqCallback);
      if (detectedStopSeq) {
        _n_past -= partialStopSeqMatchTokens.size();
        for (auto& engine : _engine) {
          // remove stop seq tokens from KV$
          if (!engine.second->removeTokenCheckpoint(partialStopSeqMatchTokens.size())) {
            return Dialog::abort("Removal of stop sequence tokens from token checkpoint failed. " +
                                     engine.second->error(),
                                 callback);
          }
          if (!engine.second->updateKV(_n_past))
            return Dialog::abort(
                "Removal of stop sequence tokens from KV cache failed. " + engine.second->error(),
                callback);
        }
        clearPartialStopSeqMatches();
      }
      return returnVal;
    }
    addPromptTokenHistory(p_vec);
    return process(p_vec, callback);
  }
  addPromptTokenHistory(p_vec);
  return process(p_vec, __no_response);
}

void Dialog::addPromptTokenHistory(std::vector<int32_t>& tokenIds) {
  for (auto& [type, sampler] : _sampler) {
    if (type == "primary") {
      sampler->updateSampledTokenHistory(tokenIds);
    }
  }
}

bool Dialog::query(const std::vector<uint32_t>& input,
                   Sentence::Code scode,
                   qualla::DialogCallback& callback) {
  std::vector<int32_t> p_vec;  // prompt tokens
  p_vec.reserve(1024);

  // always reset before start.
  m_rewindAtBoundary = false;
  _tokenizer->cleanUp();  // clean dangling history of the previous queries.

  if (scode == Sentence::COMPLETE || scode == Sentence::BEGIN) {
    // Reset prompt/gen counts for new query
    _n_prompt             = 0;
    _n_generated          = 0;
    _n_previous_prompt    = 0;
    _n_previous_generated = 0;

    if (_last_tok >= 0 && !detectedStopSeq)  // avoid putting stop sequence in query
      p_vec.push_back(_last_tok);
    detectedStopSeq = false;

    // Add EOS explicitly if the last query was aborted prematurely.
    if (_n_queries && _last_tok != _ctx->eos_tok()) {
      p_vec.push_back(_ctx->eos_tok());
    }
    // Add BOS
    if (_ctx->bos_tok() >= 0) {
      p_vec.push_back(_ctx->bos_tok());
    }
  }

  if (scode != Sentence::RESUME) {
    if (m_processState != NO_RESUME) {
      return Dialog::abort("Need to resume a paused query. ", callback);
    }
    p_vec.insert(p_vec.end(), input.begin(), input.end());
  } else {
    if (!supportsPauseResume()) {
      return Dialog::abort("Pause/Resume is not supported on this dialog. ", callback);
    }
    if (m_processState == NO_RESUME) {
      return Dialog::abort("Cannot resume a query which is not paused. ", callback);
    }
    p_vec = m_unprocessedTokens;
    __DEBUG("Resuming dialog-query with: {}", p_vec);
  }
  __DEBUG("dialog-tokens: {} {}", _ctx->name(), p_vec);

  _n_queries++;

  if (scode == Sentence::COMPLETE || scode == Sentence::END || scode == Sentence::RESUME) {
    addPromptTokenHistory(p_vec);
    return process(p_vec, callback);
  }

  DialogCallback callback_return_token(QUALLA_CALLBACK_TYPE_TOKEN);
  *(callback_return_token.getTokenCbFunc()) = __no_response_token;
  addPromptTokenHistory(p_vec);
  return process(p_vec, callback_return_token);
}

bool Dialog::query(std::vector<uint8_t>& embedding_vectors,
                   Sentence::Code scode,
                   T2ECallback t2eCallback,
                   Dialog::Callback callback) {
  if (t2eCallback == nullptr) {
    t2eCallback =
        m_t2eCallbacks["QNN_DATATYPE_FLOAT_32"]["QNN_DATATYPE_FLOAT_32"];  // default callback
    if (lutDataType != "QNN_DATATYPE_FLOAT_32") {
      t2eCallback = m_t2eCallbacks[lutDataType][inputDataType];
    }
  }
  // always reset before start.
  m_rewindAtBoundary = false;
  _tokenizer->cleanUp();  // clean dangling history of the previous queries.
  _n_queries++;
  if (scode == Sentence::RESUME) {
    if (!supportsPauseResume()) {
      return Dialog::abort("Pause/Resume is not supported on this dialog. ", callback);
    }
    if (m_processState == NO_RESUME) {
      return Dialog::abort("Cannot resume a query which is not paused. ", callback);
    }
    embedding_vectors = m_unprocessedEmbedding;
    __DEBUG("Resuming dialog-query with: {}", embedding_vectors);
  } else {
    if (m_processState != NO_RESUME) {
      return Dialog::abort("Need to resume a paused query. ", callback);
    }
  }
  if (scode == Sentence::COMPLETE || scode == Sentence::END || scode == Sentence::RESUME) {
    // Reset prompt/gen counts for new query
    _n_prompt             = 0;
    _n_generated          = 0;
    _n_previous_prompt    = 0;
    _n_previous_generated = 0;

    detectedStopSeq = false;
    if (!_stop_sequence.empty()) {
      _stop_sequence.reset();
      Dialog::Callback stopSeqCallback = [&](const std::string& str, Sentence::Code c) {
        return getStopSeqCallback(str, c, callback);
      };
      auto returnVal = process(
          embedding_vectors, t2eCallback, stopSeqCallback);  // process(p_vec, stopSeqCallback);
      if (detectedStopSeq) {
        _n_past -= partialStopSeqMatchTokens.size();
        if (!removeStopSeqFromKV())
          return Dialog::abort("Removal of stop sequence tokens from KV cache failed. ", callback);
        clearPartialStopSeqMatches();
      }
      return returnVal;
    }
    return process(embedding_vectors, t2eCallback, callback);
  }
  // Only process, no output
  return process(
      embedding_vectors, t2eCallback, [&](const std::string&, Sentence::Code) { return false; });
}

bool Dialog::query(std::vector<uint8_t>& embedding_vectors,
                   Sentence::Code scode,
                   T2ECallback t2eCallback,
                   qualla::DialogCallback& callback) {
  if (t2eCallback == nullptr) {
    t2eCallback =
        m_t2eCallbacks["QNN_DATATYPE_FLOAT_32"]["QNN_DATATYPE_FLOAT_32"];  // default callback
    if (lutDataType != "QNN_DATATYPE_FLOAT_32") {
      t2eCallback = m_t2eCallbacks[lutDataType][inputDataType];
    }
  }
  // always reset before start.
  m_rewindAtBoundary = false;
  _tokenizer->cleanUp();  // clean dangling history of the previous queries.
  _n_queries++;
  if (scode == Sentence::RESUME) {
    if (!supportsPauseResume()) {
      return Dialog::abort("Pause/Resume is not supported on this dialog. ", callback);
    }
    if (m_processState == NO_RESUME) {
      return Dialog::abort("Cannot resume a query which is not paused. ", callback);
    }
    embedding_vectors = m_unprocessedEmbedding;
    __DEBUG("Resuming dialog-query with: {}", embedding_vectors);
  } else {
    if (m_processState != NO_RESUME) {
      return Dialog::abort("Need to resume a paused query. ", callback);
    }
  }
  if (scode == Sentence::COMPLETE || scode == Sentence::END || scode == Sentence::RESUME) {
    // Reset prompt/gen counts for new query
    _n_prompt             = 0;
    _n_generated          = 0;
    _n_previous_prompt    = 0;
    _n_previous_generated = 0;

    return process(embedding_vectors, t2eCallback, callback);
  }
  // Only process, no output
  DialogCallback callback_return_token(QUALLA_CALLBACK_TYPE_TOKEN);
  *(callback_return_token.getTokenCbFunc()) = __no_response_token;
  return process(embedding_vectors, t2eCallback, callback_return_token);
}

bool Dialog::removeStopSeqFromKV() {
  for (auto& engine : _engine) {
    // remove stop seq tokens from KV$
    if (!engine.second->updateKV(_n_past)) return false;
  }
  return true;
}

bool Dialog::prime(const std::string& str) {
  bool r = query(str, Sentence::COMPLETE, __no_response);

  // End with EOS as we want the primer to be self-contained
  _last_tok = _ctx->eos_tok();

  return r;
}

bool Dialog::save(const std::string& o_name) {
  Timer start;

  // Save using session name unless override is provided
  std::string name   = o_name.empty() ? _ctx->name() : o_name;
  fs::path save_path = name;

  if (!_n_past) {
    __ERROR("dialog-save: {} : nothing to save yet", name);
    return false;
  }

  __INFO("dialog-save: saving as {} {}", name, save_path.string());

  if (!fs::exists(save_path) && !fs::create_directories(save_path)) {
    __ERROR("dialog-save: {} : failed to create cache directory", name);
    return false;
  }

  // Save Dialog state
  qualla::json j{{"n-past", _n_past},
                 {"n-prompt", _n_prompt},
                 {"n-generated", _n_generated},
                 {"n-queries", _n_queries},
                 {"last-tok", _last_tok},
                 {"process-state", m_processState},
                 {"unprocessed-tokens-size", m_unprocessedTokens.size()},
                 {"unprocessed-embedding-size", m_unprocessedEmbedding.size()}};
  {
    fs::path p = save_path / "dialog.json";
    std::ofstream f(p);
    f << j;
  }
  {
    fs::path q = save_path / "unprocessed-data";
    std::ofstream g(q, std::ios::binary);
    g.write(reinterpret_cast<char*>(m_unprocessedTokens.data()),
            static_cast<std::streamsize>(m_unprocessedTokens.size() * sizeof(int32_t)));
    g.write(reinterpret_cast<char*>(m_unprocessedEmbedding.data()),
            static_cast<std::streamsize>(m_unprocessedEmbedding.size() * sizeof(uint8_t)));
    g.close();
    g.clear();
  }

  // Save Engines (mandatory)
  for (auto& e : _engine) {
    if (!e.second->save(name)) {
      __ERROR("dialog-save: {} : unable to save {} engine. {}", name, e.first, e.second->error());
      return false;
    }
  }

  // Save Samplers (optional)
  for (auto& s : _sampler) {
    if (!s.second->save(name)) {
      __WARN("dialog-save: {} : unable to save {} sampler", name, s.first);
    }
  }

  _kpis.save.update(start.elapsed_usec());

  return true;
}

bool Dialog::restore(const std::string& o_name) {
  Timer start;

  // Restore using session name unless override is provided
  std::string name      = o_name.empty() ? _ctx->name() : o_name;
  fs::path restore_path = name;

  __INFO("dialog-restore: restoring from {} {}", name, restore_path.string());

  // Try to restore the Dialog state (optional)
  // If this fails we reset everything and try to restore the engine.
  qualla::json j{};
  {
    fs::path p = restore_path / "dialog.json";
    if (fs::exists(p)) {
      std::ifstream f(p);
      j = qualla::json::parse(f);
    } else {
      __DEBUG("dialog-restore: {} : internal state not restored", name);
    }
  }

  using qc                        = qualla::Config;
  _n_past                         = qc::optional<uint32_t>(j, "n-past", 0);
  _n_prompt                       = qc::optional<uint32_t>(j, "n-prompt", 0);
  _n_generated                    = qc::optional<uint32_t>(j, "n-generated", 0);
  _n_queries                      = qc::optional<uint32_t>(j, "n-queries", 1);
  _last_tok                       = qc::optional<int32_t>(j, "last-tok", _ctx->eos_tok());
  uint8_t processState            = qc::optional<uint8_t>(j, "process-state", 0);
  size_t unprocessedTokensSize    = qc::optional<size_t>(j, "unprocessed-tokens-size", 0);
  size_t unprocessedEmbeddingSize = qc::optional<size_t>(j, "unprocessed-embedding-size", 0);

  m_processState = static_cast<ProcessState>(processState);
  m_unprocessedTokens.resize(unprocessedTokensSize);
  m_unprocessedEmbedding.resize(unprocessedEmbeddingSize);
  {
    fs::path q = restore_path / "unprocessed-data";
    if (fs::exists(q)) {
      std::ifstream g(q, std::ios::binary);
      g.read(reinterpret_cast<char*>(m_unprocessedTokens.data()),
             static_cast<std::streamsize>(m_unprocessedTokens.size() * sizeof(int32_t)));
      g.read(reinterpret_cast<char*>(m_unprocessedEmbedding.data()),
             static_cast<std::streamsize>(m_unprocessedEmbedding.size() * sizeof(uint8_t)));
      g.close();
      g.clear();
    } else {
      __DEBUG("dialog-restore: {} : internal state not restored", name);
    }
  }

  // Restore Engines (mandatory)
  for (auto& e : _engine) {
    uint32_t n = e.second->restore(name);
    if (!n) {
      __ERROR(
          "dialog-restore: {} : unable to restore {} engine. {}", name, e.first, e.second->error());
      return false;
    }

    // Restore n_past from the engine state
    if (_n_past && n != _n_past) {
      __WARN("dialog-restore: {} : n-past mismatch : {} engine {} intern {}",
             name,
             e.first,
             _n_past,
             n);
      // Keep the smaller number
      _n_past = std::min(n, _n_past);
    } else
      _n_past = n;
  }

  // Restore Samplers (optional)
  for (auto& s : _sampler) {
    if (!s.second->restore(name)) {
      __WARN("dialog-restore: {} : unable to restore {} sampler", name, s.first);
    }
  }

  _kpis.reset();
  _kpis.restore.update(start.elapsed_usec());

  return true;
}

void Dialog::reset() {
  __INFO("dialog-reset: {}", _ctx->name());

  _n_past               = 0;
  _n_prompt             = 0;
  _n_generated          = 0;
  _n_queries            = 0;
  _last_tok             = -1;
  _n_previous_prompt    = 0;
  _n_previous_generated = 0;
  m_processState        = NO_RESUME;
  m_unprocessedEmbedding.clear();
  m_unprocessedTokens.clear();

  _kpis.reset();
  m_perfProfile = m_defaultPerfProfile;
  // Reset Engines and Samplers
  for (auto& e : _engine) {
    e.second->setPerfProfile(m_perfProfile);
    e.second->reset();
  }
  for (auto& s : _sampler) s.second->reset();

  State::clear();
}

// Dialog KPIs helpers

// Get latest KPIs
Dialog::KPIs& Dialog::kpis() {
  // Update TPS
  if (_n_prompt) {
    float t            = _kpis.prompt.last_usec / _n_prompt;
    _kpis.tps.n_prompt = _n_prompt;
    _kpis.tps.prompt   = 1000000.0f / (t ? t : 1000000.0f);
  }

  if (_n_generated) {
    float t              = _kpis.generate.last_usec / _n_generated;
    _kpis.tps.n_generate = _n_generated;
    _kpis.tps.generate   = 1000000.0f / (t ? t : 1000000.0f);
  }

  // We could synthesize more KPIs from from other layers (engine, sampler, etc)
  return _kpis;
}

std::string Dialog::KPIs::dump(std::string_view sep) const {
  return fmt::format(
      "init:[{}]{}prompt:[{}]{}generate:[{}]{}save:[{}]{}restore:[{}]{} tps-prompt:{:.2f} "
      "tps-generate:{:.2f}",
      init.dump(),
      sep,
      prompt.dump(),
      sep,
      generate.dump(),
      sep,
      save.dump(),
      sep,
      restore.dump(),
      sep,
      tps.prompt,
      tps.generate);
}

void Dialog::KPIs::reset() {
  prompt.reset();
  generate.reset();
  save.reset();
  restore.reset();
  tps.prompt   = 0.0f;
  tps.generate = 0.0f;
}

// Create API
std::unique_ptr<Dialog> Dialog::create(std::shared_ptr<Env> env,
                                       const std::string& name,
                                       const qualla::json& conf) {
  const std::string type = qualla::Config::optional<std::string>(conf, "type", BasicDialog::TYPE);

  if (type == BasicDialog::TYPE) {
    return std::make_unique<BasicDialog>(env, name, conf);
  }
  if (type == EagletDialog::TYPE) {
    return std::make_unique<EagletDialog>(env, name, conf);
  }
  if (type == KvShareDialog::TYPE) {
    return std::make_unique<KvShareDialog>(env, name, conf);
  }
  if (type == LhdDecDialog::TYPE) {
    return std::make_unique<LhdDecDialog>(env, name, conf);
  }
  if (type == MultiStreamDialog::TYPE) {
    return std::make_unique<MultiStreamDialog>(env, name, conf);
  }
  if (type == SpecDecDialog::TYPE) {
    return std::make_unique<SpecDecDialog>(env, name, conf);
  }
  if (type == SelfSpecDecDialog::TYPE) {
    return std::make_unique<SelfSpecDecDialog>(env, name, conf);
  }

  throw std::runtime_error(type + ": dialog not found");
}

std::unique_ptr<Dialog> Dialog::create(std::shared_ptr<Env> env,
                                       const std::string& name,
                                       std::istream& json_stream) {
  return create(env, name, json::parse(json_stream));
}

std::unique_ptr<Dialog> Dialog::create(std::shared_ptr<Env> env,
                                       const std::string& name,
                                       const fs::path& json_path) {
  if (!fs::exists(json_path)) {
    throw std::runtime_error(json_path.string() + ": file does not exist");
  }
  std::ifstream ifs(json_path);
  return create(env, name, ifs);
}

std::vector<std::string> Dialog::list() {
  static const std::vector<std::string> s_dialogTypes{BasicDialog::TYPE,
                                                      EagletDialog::TYPE,
                                                      KvShareDialog::TYPE,
                                                      LhdDecDialog::TYPE,
                                                      MultiStreamDialog::TYPE,
                                                      SpecDecDialog::TYPE,
                                                      SelfSpecDecDialog::TYPE};

  return s_dialogTypes;
}

bool Dialog::applyLoraAdapter(std::string lora_adapter_name, std::string engine_role) {
  if (_engine.count(engine_role) == 0) {
    __ERROR(
        "Dialog::applyLoraAdapter: specified {} engine type is invalid for apply LoRA adapters.",
        engine_role);
    return false;
  }
  _kpis.lora.last_usec  = 0;
  _kpis.lora.total_usec = 0;
  Timer start;
  if (_sharedEngine.contains(engine_role) && _sharedEngine[engine_role]->busy()) {
    __ERROR("dialog-applyLoraAdapter: failed for {} as shared engine {} is busy",
            lora_adapter_name,
            engine_role);
    return false;
  }
  _engine[engine_role]->busy(true);
  if (!_engine[engine_role]->applyLoraAdapter(lora_adapter_name)) {
    __WARN("dialog-applyLoraAdapter: failed for {}", lora_adapter_name);
    return false;
  }
  if (_encoder && _encoder->type() == "lut") {
    calculateRequantEncodings();
  }
  _engine[engine_role]->busy(false);
  _kpis.lora.update(start.elapsed_usec());
  return true;
}

bool Dialog::applyLoraStrength(std::string tensor_name, float tensor_val, std::string engine_role) {
  if (_engine.count(engine_role) == 0) {
    __ERROR("Dialog::applyLoraAdapter: specified {} engine type is invalid for set LoRA strength.",
            engine_role);
    return false;
  }
  if (_sharedEngine.contains(engine_role) && _sharedEngine[engine_role]->busy()) {
    __ERROR(
        "dialog-setStrength: failed for {} as shared engine {} is busy", tensor_name, engine_role);
    return false;
  }
  _engine[engine_role]->busy(true);
  if (!_engine[engine_role]->applyLoraStrength(tensor_name, tensor_val)) {
    __WARN("dialog-applyLoraStrength: failed for {}", tensor_name);
    return false;
  }
  _engine[engine_role]->busy(false);
  return true;
}

bool Dialog::kvRewindPrefixMatch(std::vector<int32_t>& p_vec) {
  _kpis.prompt.last_usec  = 0;
  _kpis.prompt.total_usec = 0;
  Timer start;
  for (auto& e : _engine) {
    auto [rewind_token_index, nextToken] = e.second->rewindKVCacheToPrefixMatch(p_vec, _n_past);
    if (rewind_token_index != 0) {
      auto first = p_vec.begin();
      auto last  = p_vec.begin() + rewind_token_index;
      p_vec.erase(first, last);
      /* because the index is rewind,remainingy prompt already conisdered as
        generated for the KPIs formula to satisfy*/
      _n_prompt = rewind_token_index;
      if (p_vec.size() == 0 && nextToken != -1) {
        m_rewindAtBoundary = true;
        p_vec.push_back(nextToken);
      }
    }
  }
  return true;
}

void Dialog::setStopSequence(const qualla::json& newStopSeqsJson) {
  const std::vector<std::string>& newStopSequences =
      qualla::Config::optional<std::vector<std::string>>(newStopSeqsJson, "stop-sequence", {});
  _stop_sequence.clear();
  _stop_sequence.build_trie(newStopSequences);
}

bool Dialog::getStopSeqCallback(const std::string& str,
                                Sentence::Code c,
                                Dialog::Callback callback) {
  // Check for stop sequence and end inference when stop sequence is found
  auto [stopSeqStatus, stopSeqIndex] = _stop_sequence.process_next_string(str);
  if (stopSeqStatus == SequenceMatchTrie::MatchType::COMPLETE_MATCH) {
    detectedStopSeq = true;
    addPartialStopSeqMatches(str, stopSeqIndex);
    // If the stop sequence is subpart of current or a previous partial match token,
    // output the part of the token occurring before stop sequence
    callback(partialStopSeqMatchTokens[0].substr(0, partialStopSeqMatchIndexes[0]),
             Sentence::CONTINUE);
    callback("", Sentence::END);  // Match is complete. Stop emit sequences.
    return false;
  } else if (stopSeqStatus == SequenceMatchTrie::MatchType::PARTIAL_MATCH) {
    // Hold str in partialStopSeqMatchTokens without outputting it
    if (partialStopSeqMatchTokens.size() > 0 && stopSeqIndex > 0) {
      // New partial match detected. Old partial matchs failed. So, fully output old partial match
      // tokens.
      std::string accumulatedStr = accumulatePartialStopSeqMatches();
      callback(accumulatedStr, c);
      clearPartialStopSeqMatches();
    }
    addPartialStopSeqMatches(str, stopSeqIndex);
    if (c == Sentence::END) {
      // The partial matches haven't reached COMPLETE_MATCH even at the end of sentence.
      // So, output all held tokens.
      std::string accumulatedStr = accumulatePartialStopSeqMatches();
      auto returnValue           = callback(accumulatedStr, Sentence::CONTINUE);
      callback("", Sentence::END);
      clearPartialStopSeqMatches();
      return returnValue;
    }
    return callback("", c);
  } else {
    // If there were partial matches earlier and the current token caused a NO_MATCH,
    // output the previous partial matches as well.
    std::string accumulatedStr = accumulatePartialStopSeqMatches();
    auto returnValue           = callback(accumulatedStr + str, c);
    clearPartialStopSeqMatches();
    return returnValue;
  }
}
bool Dialog::setOemKey(const std::string& oemKey) {
  for (auto& e : _engine) {
    if (!e.second->setOemkey(oemKey)) {
      __ERROR("Dialog::setOemKey: unable to set OEM key for engine.error = {}", e.second->error());
      return false;
    }
  }
  return true;
}

bool Dialog::setExecutionPriority(std::string engine_role, uint32_t exeuctionPriority) {
  if (_engine.count(engine_role) == 0) {
    __ERROR(
        "Dialog::setExecutionPriority: specified {} engine type is invalid for execution priority "
        "setting.",
        engine_role);
    return false;
  }
  auto& engine = *_engine[engine_role];
  if (!engine.setExecutionPriority(exeuctionPriority)) {
    __WARN("Dialog::setExecutionPriority: failed for {}", exeuctionPriority);
    return false;
  }

  return true;
}

std::shared_ptr<Engine> Dialog::getEngine(const std::string& engineRole) {
  Timer start;
  if (_engine.count(engineRole) == 0) {
    __ERROR("Dialog::getEngine: specified {} engine type is invalid.", engineRole);
    return {};
  }

  auto engine = _engine[engineRole];
  _kpis.getEngine.reset();
  _kpis.getEngine.update(start.elapsed_usec());
  return engine;
}

bool Dialog::bindEngine(const std::string& engineRole, std::shared_ptr<Engine> engine) {
  Timer start;
  if (engine && engine->isBound()) {
    __ERROR("Dialog::bindEngine: failed to bind already bounded engine");
    return false;
  }

  if (_engine.count(engineRole) == 0) {
    __ERROR("Dialog::bindEngine: specified {} engine type is invalid for binding.", engineRole);
    return false;
  }

  _engine[engineRole]->unBound();
  engine->bound();
  _engine[engineRole] = engine;
  _kpis.bindEngine.reset();
  _kpis.bindEngine.update(start.elapsed_usec());
  return true;
}

void Dialog::validate() const {
  if (!this->supportsLongContext()) {
    for (auto& engine : _engine) {
      if (engine.second->isLongContextEnabled()) {
        throw std::runtime_error("Cannot enable Long Context on this dialog.");
      }
    }
  }
}

bool Dialog::markEnginesBusy() {
  for (auto& [_, engine] : _sharedEngine) {
    if (engine->busy()) {
      __WARN("All engines are not free.");
      return false;
    }
    engine->busy(true);
  }
  return true;
}

void Dialog::markEnginesFree() {
  for (auto& [_, engine] : _sharedEngine) {
    engine->busy(false);
  }
}

bool Dialog::bindSharedEngine(const std::string& engineRole, std::shared_ptr<Engine> engine) {
  Timer start;
  if (engine->busy()) {
    return false;
  }
  engine->busy(true);
  _sharedEngine[engineRole] = engine;
  if (m_loraConfig.contains(engineRole))
    _engineState[engineRole] = std::make_shared<EngineState>(_env, m_loraConfig[engineRole]);
  else
    _engineState[engineRole] = std::make_shared<EngineState>(_env);

  // add reference to main engine object as well.
  _engine[engineRole] = _sharedEngine[engineRole];
  if (!applyEnginesState()) {
    __ERROR("Error: Failed to share the engine.");
    return false;
  }
  _kpis.bindEngine.reset();
  _kpis.bindEngine.update(start.elapsed_usec());
  engine->busy(false);
  return true;
}

void Dialog::bindSharedEngines(std::unordered_map<std::string, std::shared_ptr<Engine>>& engines) {
  for (auto [engineRole, engine] : engines) {
    if (!bindSharedEngine(engineRole, engine) || engine->busy()) {
      throw std::runtime_error("Error: Failed to bind engine.");
    }
  }
}

bool Dialog::applyEnginesState() {
  Timer start;
  if (_sharedEngine.size() != _engineState.size()) {
    std::string err = "Error: expected same number of engine States as engine, but found " +
                      std::to_string(_engineState.size()) + "engine states, for " +
                      std::to_string(_sharedEngine.size()) + "engines";
    throw std::runtime_error(err);
  }

  for (auto& [role, engineState] : _engineState) {
    if (true != _sharedEngine[role]->applyEngineState(engineState)) {
      __ERROR("Error: Failed to update engine states");
      return false;
    }
  }

  // update the state from the engine back to dialog
  // refresh the state after engine is done with it work.
  for (auto& [role, engineState] : _engineState) {
    if (true != engineState->update(_sharedEngine[role]->getEngineState())) {
      __ERROR("Error: Failed to update engine states");
      return false;
    }
  }

  completeInit();

  // early exit in case of CPU engine
  if (!_engine.empty() && _engine.contains("primary")) {
    if (!strcmp(_engine["primary"]->type().c_str(), "qnn-cpu")) {
      _kpis.applyEngineState.reset();
      _kpis.applyEngineState.update(start.elapsed_usec());
      return true;
    }
  }

  if (_encoder && _encoder->type() == "lut") calculateRequantEncodings();
  // Todo: In cross dialog some parameters needs to be reset. Find a better way to reset these
  // params. In SSD these are set per query so no need to reset.
  if (!_engine.empty() && _engine.contains("primary")) {
    auto data = _engine["primary"]->get();
    if (data.contains("kv-prefix-skip")) {
      data["kv-prefix-skip"] = 0;
    }
    if (data.contains("kv-prefix-offset")) {
      data["kv-prefix-offset"] = 0;
    }
    _engine["primary"]->set(data);
  }

  _kpis.applyEngineState.reset();
  _kpis.applyEngineState.update(start.elapsed_usec());
  return true;
}

}  // namespace qualla
