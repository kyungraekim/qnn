//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef TEXT_GENERATOR_HPP
#define TEXT_GENERATOR_HPP
#include <memory>
#include <vector>

#include "GeniePipeline.h"
#include "Node.hpp"

namespace genie {
namespace pipeline {

class Pipeline;

class TextGenerator final : public Node {
 public:
  TextGenerator(qualla::json config,
                std::shared_ptr<ProfileStat> profileStat,
                std::shared_ptr<genie::log::Logger> logger = nullptr);

  int32_t bindPipeline(Pipeline& pipeline);
  int32_t execute(void* userData, std::shared_ptr<ProfileStat> profileStat);
  int32_t save(const std::string&);
  int32_t restore(const std::string&);
  void reset();
  int32_t setPriority(std::string engine, GeniePipeline_Priority_t priority);
  int32_t setOemkey(const std::string& oemKey);

  // Set input data for the node
  int32_t setTextInputData(GenieNode_IOName_t nodeIOName,
                           const char* txt,
                           std::shared_ptr<ProfileStat> profileStat);
  int32_t setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                const void* embedding,
                                size_t embeddingSize,
                                std::shared_ptr<ProfileStat> profileStat);

  int32_t setTextOutputCallback(GenieNode_IOName_t nodeIOName,
                                GenieNode_TextOutput_Callback_t callback);

  int32_t applyLora(std::string loraAdapterName,
                    std::string engine,
                    std::shared_ptr<ProfileStat> profileStat);
  int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);

  GenieEngine_Handle_t getEngineHandle(const std::string& engineRole,
                                       std::shared_ptr<ProfileStat> profileStat);
  int32_t bindEngine(const std::string& engineRole,
                     std::shared_ptr<Engine> engine,
                     std::shared_ptr<ProfileStat> profileStat);
  GenieSampler_Handle_t getSamplerHandle();
  GenieTokenizer_Handle_t getTokenizerHandle();

 private:
  std::string m_queryString = "";
  std::shared_ptr<Dialog> m_generator;
  size_t m_accumulatorSize = 0;
  GenieNode_TextOutput_Callback_t m_textOutputCallback;
};

}  // namespace pipeline
}  // namespace genie
#endif  // TEXT_GENERATOR_HPP