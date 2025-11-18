//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "BufferUtils.hpp"
#include "QnnTypeMacros.hpp"
#include "QnnTypeUtils.hpp"

namespace aiswutility {

#if !defined(__arm__)
uint32_t calculateByteLength(const std::vector<uint32_t>& dims, const Qnn_DataType_t dataType) {
  uint32_t length = static_cast<uint32_t>(getDataTypeSize(dataType));
  length *= calculateElementCount(dims);
  return length;
}
#endif
size_t calculateByteLength(const std::vector<size_t>& dims, const Qnn_DataType_t dataType) {
  size_t length = static_cast<size_t>(getDataTypeSize(dataType));
  length *= calculateElementCount(dims);
  return length;
}

#if !defined(__arm__)
uint32_t calculateByteLength(const std::vector<uint32_t>& dims,
                             const Qnn_DataType_t dataType,
                             bool& success) {
  uint32_t length = calculateByteLength(dims, dataType);
  if (length == 0) {  // half byte sizes will have their length be 0. Unrecognized datatypes also
                      // have a length of 0.
    success = false;
    return length;
  }
  success = true;
  return length;
}
#endif
size_t calculateByteLength(const std::vector<size_t>& dims,
                           const Qnn_DataType_t dataType,
                           bool& success) {
  size_t length = calculateByteLength(dims, dataType);
  if (length == 0) {  // half byte sizes will have their length be 0. Unrecognized datatypes also
                      // have a length of 0.
    success = false;
    return length;
  }
  success = true;
  return length;
}

#if !defined(__arm__)
float calculateByteExactLength(const std::vector<uint32_t>& dims, const Qnn_DataType_t dataType) {
  return calculateElementCount(dims) * getDataTypeSize(dataType);
}
#endif
float calculateByteExactLength(const std::vector<size_t>& dims, const Qnn_DataType_t dataType) {
  return calculateElementCount(dims) * getDataTypeSize(dataType);
}

#if !defined(__arm__)
uint32_t calculateElementCount(const std::vector<uint32_t>& dims) {
  return static_cast<uint32_t>(
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()));
}
#endif
size_t calculateElementCount(const std::vector<size_t>& dims) {
  return static_cast<size_t>(
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()));
}

uint64_t queryTensorSize(Qnn_Tensor_t tensor) {
  uint32_t* const dims      = QNN_TENSOR_GET_DIMENSIONS(tensor);
  const uint32_t dataFormat = QNN_TENSOR_GET_DATA_FORMAT(tensor);
  return (uint64_t)getBufferSize(dims, dataFormat);
}

uint32_t getBufferSize(const uint32_t* dims, const uint32_t dataFormat) {
  uint32_t height     = dims[1];
  uint32_t width      = dims[2];
  uint32_t bufferSize = 0;
  switch (dataFormat) {
    case QNN_TENSOR_DATA_FORMAT_UBWC_RGBA8888: {
      // Metadata
      bufferSize = DATA_FMT_ALIGN(
          DATA_FMT_ALIGN(((width + 16 - 1) / 16), 64) * DATA_FMT_ALIGN(((height + 4 - 1) / 4), 16),
          4096);
      // Compressed
      bufferSize +=
          DATA_FMT_ALIGN(DATA_FMT_ALIGN(width, 64) * 4 * DATA_FMT_ALIGN(height, 16), 4096);
      break;
    }
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12_UV: {
      /* adjust h and w to account for full tensor size*/
      width *= 2;
      height *= 2;
    }
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12:
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12_Y: {
      // Plane 0 (Y plane)
      // Metadata
      bufferSize = DATA_FMT_ALIGN(
          DATA_FMT_ALIGN(((width + 32 - 1) / 32), 64) * DATA_FMT_ALIGN(((height + 8 - 1) / 8), 16),
          4096);
      // Compressed
      bufferSize += DATA_FMT_ALIGN(DATA_FMT_ALIGN(width, 128) * DATA_FMT_ALIGN(height, 32), 4096);
      // Plane 1 (UV plane)
      // Metadata
      bufferSize += DATA_FMT_ALIGN(DATA_FMT_ALIGN(((width / 2 + 16 - 1) / 16), 64) *
                                       DATA_FMT_ALIGN(((height / 2 + 8 - 1) / 8), 16),
                                   4096);
      // Compressed
      bufferSize +=
          DATA_FMT_ALIGN(DATA_FMT_ALIGN(width / 2, 64) * 2 * DATA_FMT_ALIGN(height / 2, 32), 4096);
      break;
    }
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_UV: {
      /* adjust h and w to account for full tensor size*/
      width *= 2;
      height *= 2;
    }
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R:
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_Y: {
      // Plane 0 (Y plane)
      // Metadata
      bufferSize = DATA_FMT_ALIGN(
          DATA_FMT_ALIGN(((width + 64 - 1) / 64), 64) * DATA_FMT_ALIGN(((height + 4 - 1) / 4), 16),
          4096);
      // Compressed
      bufferSize += DATA_FMT_ALIGN(DATA_FMT_ALIGN(width, 256) * DATA_FMT_ALIGN(height, 16), 4096);
      // Plane 1 (UV plane)
      // Metadata
      bufferSize += DATA_FMT_ALIGN(DATA_FMT_ALIGN(((width / 2 + 32 - 1) / 32), 64) *
                                       DATA_FMT_ALIGN(((height / 2 + 4 - 1) / 4), 16),
                                   4096);
      // Compressed
      bufferSize +=
          DATA_FMT_ALIGN(DATA_FMT_ALIGN(width / 2, 128) * 2 * DATA_FMT_ALIGN(height / 2, 16), 4096);
      break;
    }
    default:
      return bufferSize;
  }
  return bufferSize;
}

}  // namespace aiswutility
