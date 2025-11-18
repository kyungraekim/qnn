//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>
#include <vector>

#include "Exception.hpp"
#include "ImageEncoder.hpp"
#include "Pipeline.hpp"

using namespace genie;

pipeline::ImageEncoder::ImageEncoder(qualla::json config,
                                     std::shared_ptr<ProfileStat> profileStat,
                                     std::shared_ptr<genie::log::Logger> logger)
    : Node(config) {
  for (auto item : m_config.items()) {
    qualla::json embeddingConfig;
    embeddingConfig["embedding"]         = item.value();
    embeddingConfig["embedding"]["type"] = "image-encoder";
    if (embeddingConfig["embedding"].contains("engine")) {
      qualla::json& embeddingEngineConfig = embeddingConfig["embedding"]["engine"];
      // Set defaults for Embedding QnnHtp backend
      if (embeddingEngineConfig.contains("backend")) {
        if (embeddingEngineConfig["backend"].contains("QnnHtp")) {
          embeddingEngineConfig["backend"]["QnnHtp"]["pooled-output"]    = false;
          embeddingEngineConfig["backend"]["QnnHtp"]["disable-kv-cache"] = true;
        }
      }
    }
    Embedding::validateEmbeddingConfig(embeddingConfig["embedding"], false);
    m_encoder = std::make_shared<genie::Embedding>(embeddingConfig, profileStat, logger);
  }

  std::unordered_set<std::string> inputNames;
  m_encoder->getInputNames(inputNames);
  for (auto& name : inputNames) {
    if (m_ioNameTable.find(name) == m_ioNameTable.end()) {
      throw Exception(GENIE_STATUS_ERROR_GENERAL,
                      "ImageEncoder meet unsupported input layer of model");
    }

    m_inputIOMap[m_ioNameTable[name]] = name;
  }
}

int32_t pipeline::ImageEncoder::setEmbeddingOutputCallback(
    GenieNode_IOName_t nodeIOName, GenieNode_EmbeddingOutputCallback_t callback) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT) {
    throw Exception(
        GENIE_STATUS_ERROR_GENERAL,
        "setEmbeddingOutputCallback can only be set for GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT");
  }
  m_embeddingOutputCallback = callback;
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::ImageEncoder::setImageInputData(GenieNode_IOName_t nodeIOName,
                                                  const void* imageData,
                                                  size_t imageSize,
                                                  std::shared_ptr<ProfileStat> /*profileStat*/) {
  if (m_inputIOMap.find(nodeIOName) == m_inputIOMap.end()) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Unsupported IOName in setImageInputData");
  }
  if (imageData == nullptr) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "setImageInputData get nullptr imageData");
  }
  if (imageSize == 0) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "setImageInputData get imageSize 0");
  }

  const uint8_t* dataPtr = static_cast<const uint8_t*>(imageData);
  std::string name       = m_inputIOMap[nodeIOName];
  m_input[name]          = std::vector<uint8_t>(dataPtr, dataPtr + imageSize);

  int32_t status = GENIE_STATUS_SUCCESS;
  if (m_input.size() == m_inputIOMap.size()) {
    status = m_encoder->encode(m_input, m_data, nullptr);
    m_input.clear();  // clear input buffer after encode
    if (status != GENIE_STATUS_SUCCESS) {
      throw Exception(status, "ImageEncoder::setImageInputData failed");
    }

    if (isConnected()) {
      std::string outputDataType;
      double outputScale;
      int32_t outputOffset;
      size_t outputByteWidth;
      m_encoder->getOutputQuantParam(outputDataType, outputScale, outputOffset, outputByteWidth);
      size_t numElements = m_data.size() / outputByteWidth;

      m_pipeline->m_accumulator->append(
          m_data.data(), outputDataType, outputScale, outputOffset, numElements);
    }
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::ImageEncoder::execute(void* userData, std::shared_ptr<ProfileStat>) {
  std::vector<uint32_t> dimensions;
  m_encoder->getOutputDimensions(dimensions);
  if (m_embeddingOutputCallback) {  // invoke userCallback if set
    m_embeddingOutputCallback(dimensions.data(),
                              dimensions.size(),
                              m_data.size(),
                              reinterpret_cast<void*>(m_data.data()),
                              userData);
  }
  m_data.clear();  // clear encoder buffer after callback invoked

  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::ImageEncoder::applyLora(std::string loraAdapterName,
                                          std::string engine,
                                          std::shared_ptr<ProfileStat> profileStat) {
  return m_encoder->applyLora(loraAdapterName, engine, profileStat);
}

int32_t pipeline::ImageEncoder::applyLoraStrength(std::string tensorName,
                                                  std::string engine,
                                                  float alpha) {
  return m_encoder->applyLoraStrength(tensorName, engine, alpha);
}
