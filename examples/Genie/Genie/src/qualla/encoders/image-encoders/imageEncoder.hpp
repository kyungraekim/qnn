//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qualla/detail/exports.h"
#include "qualla/encoder.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"

namespace qualla {

class ImageEncoder : public Encoder {
 public:
  static constexpr const char* TYPE = "ImageEncoder";

  ImageEncoder(std::shared_ptr<Env> env, const qualla::json& conf);
  virtual ~ImageEncoder();

  virtual bool encode(const std::unordered_map<std::string, std::vector<uint8_t>>& input_tensors,
                      std::vector<uint8_t>& image_features);

  void input_names(std::unordered_set<std::string>& inputTensorNames);

  // Get output dimensions
  void output_dimensions(std::vector<std::uint32_t>& outputDimensions);

  void outputTensorQuantParam(std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& byteWidth);

 protected:
  std::vector<std::uint32_t> _output_dimensions{};

  virtual bool process(const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
                       std::vector<uint8_t>& image_features);

  size_t _model_input_height    = 384;
  size_t _model_input_width     = 384;
  size_t _model_input_channel   = 3;
  size_t _model_input_byteWidth = 1;
};

}  // namespace qualla
