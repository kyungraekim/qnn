//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "qualla/detail/json.hpp"

namespace qualla {

struct RopeScalingParams {
  enum RopeType { DEFAULT, ROPE_LLAMA3, ROPE_LONGROPE, ROPE_QWEN2VL } rope_type = DEFAULT;

  // This should be a union, but running into compilation issues with non-trivial dtr/copy-ctr
  struct {
    double factor;
    double low_freq_factor;
    double high_freq_factor;
    int original_max_position_embeddings;
  } llama3_params{};

  struct {
    double factor;
    std::vector<double> long_factor;
    std::vector<double> short_factor;
    int original_max_position_embeddings;
  } longrope_params{};

  struct {
    uint32_t height;
    uint32_t width;
    uint32_t spatial_merge_size;
    uint32_t patch_size;
    uint32_t window_size;
  } qwen2vl_params{};

  RopeScalingParams() {}
};

struct PositionalEncoding {
  enum EncodingType : uint8_t { ROPE = 0x0, ABSOLUTE = 0x1, ALIBI = 0x2, UNDEFINED = 0xff } type;
  struct {
    int32_t dims;
    double theta;
    RopeScalingParams rope_scaling;
  } rope_params{};

  PositionalEncoding() { type = ROPE; }
};

struct LongContextParams {
  enum Mode : uint8_t { DISABLED = 0, SLIDING_WINDOW = 1, KEYDIFF = 2 } mode = DISABLED;

  int32_t sink_tokens{0};
  int32_t update_frequency{128};
  int32_t window_size{0};
  std::string scoring_network;
  LongContextParams() {}
};

struct CacheGroupParams {
  std::string prefix{"past_"};

  std::string attention_mask_tensor_name{""};
  std::string cache_index_tensor_name{""};
  LongContextParams longcontext_params;
  CacheGroupParams() {}
};

using CacheGroupParamsMap = std::map<std::string, CacheGroupParams>;

// Helper functions for converting to/from jsom
void from_json(const json& j, PositionalEncoding& p);
void to_json(json& j, const PositionalEncoding& p);
void from_json(const json& j, RopeScalingParams& p);
void to_json(json& j, const RopeScalingParams& p);
void from_json(const json& j, LongContextParams& p);
void to_json(json& j, const LongContextParams& p);
void from_json(const json& j, CacheGroupParams& p);
void to_json(json& j, const CacheGroupParams& p);
void from_json(const json& j, CacheGroupParamsMap& p);
void to_json(json& j, const CacheGroupParamsMap& p);

}  // namespace qualla