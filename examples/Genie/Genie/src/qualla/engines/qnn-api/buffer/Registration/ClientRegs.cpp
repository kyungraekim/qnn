//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "QnnTypeMacros.hpp"
#include "qualla/detail/Log.hpp"
#include "qualla/detail/buffer/Registration/ClientRegs.hpp"

ClientRegs::ClientRegs(std::shared_ptr<ClientAllocator> clientAllocator)
    : m_clientAllocator(clientAllocator) {}

ClientRegs::~ClientRegs() {
  for (auto it = m_tensorToAllocIdxMap.begin(); it != m_tensorToAllocIdxMap.end();) {
    auto nxt = std::next(it);
    if (true != deregisterTensor(it->first)) {
      QNN_ERROR("Failed to deregister tensor.");
    }
    it = nxt;
  }
  m_tensorToAllocIdxMap.clear();
  m_extBufferTensors.clear();
};

void* ClientRegs::getBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor || !m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_WARN("getBuffer: received a null pointer to a tensor");
    return nullptr;
  }
  if (m_extBufferTensors.contains(tensor)) {
    return QNN_TENSOR_GET_CLIENT_BUF(tensor).data;
  }
  return m_clientAllocator->getBuffer(m_tensorToAllocIdxMap[tensor]);
}

size_t ClientRegs::getBufferSize(Qnn_Tensor_t* tensor) {
  if (!tensor || !m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_WARN("getBufferSize: received a null pointer to a tensor");
    return 0;
  }
  if (m_extBufferTensors.contains(tensor)) {
    return QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize;
  }
  return m_clientAllocator->getBufferSize(m_tensorToAllocIdxMap[tensor]);
}

int ClientRegs::getFd(Qnn_Tensor_t* /*tensor*/) {
  QNN_WARN("getFd: This is not ION memory");
  return -1;
};

bool ClientRegs::initialize() { return m_clientAllocator->initialize(); }

bool ClientRegs::registerTensor(Qnn_Tensor_t* tensor, uint64_t allocIdx) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_RAW);
  Qnn_ClientBuffer_t clientBuffer;
  clientBuffer.data     = m_clientAllocator->getBuffer(allocIdx);
  clientBuffer.dataSize = m_clientAllocator->getBufferSize(allocIdx);
  QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuffer);
  m_tensorToAllocIdxMap[tensor] = allocIdx;
  return true;
}

bool ClientRegs::deregisterTensor(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  Qnn_ClientBuffer_t temp({nullptr, 0u});
  QNN_TENSOR_SET_CLIENT_BUF(tensor, temp);
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_UNDEFINED);
  if (m_tensorToAllocIdxMap.contains(tensor)) {
    m_tensorToAllocIdxMap.erase(tensor);
  }
  return true;
}

bool ClientRegs::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  uint64_t allocIdx = m_clientAllocator->allocate(tensorDataSize);
  if (true != registerTensor(tensor, allocIdx)) {
    QNN_ERROR("mem registration failed for the clientBuffer");
    return false;
  }
  return true;
}

bool ClientRegs::mapTensorBuffer(Qnn_Tensor_t* tensor,
                                 uint64_t allocIdx,
                                 size_t /*tensorDatasize*/) {
  if (true != registerTensor(tensor, allocIdx)) {
    QNN_ERROR("mem registration failed for the clientBuffer");
    return false;
  }
  return true;
}

bool ClientRegs::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensors");
    return false;
  }
  if (m_extBufferTensors.contains(tensor)) {
    QNN_DEBUG("Tensor is using external memory with the backend.");
    return true;
  }
  if (!m_tensorToAllocIdxMap.contains(tensor)) {
    QNN_ERROR("Tensor is not registered with the backend.");
    return false;
  }
  m_clientAllocator->freeBuffer(m_tensorToAllocIdxMap[tensor]);
  if (true != deregisterTensor(tensor)) {
    QNN_ERROR("Tensor is failed to deregister.");
    return false;
  }
  return true;
}

bool ClientRegs::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    QNN_ERROR("Received nullptr");
    return false;
  }

  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_CLIENT_BUF(dest, QNN_TENSOR_GET_CLIENT_BUF(src));
  m_tensorToAllocIdxMap[dest] = m_tensorToAllocIdxMap[src];

  return true;
}

bool ClientRegs::useExternalMemory(Qnn_Tensor_t* dest, void* extMem) {
  if (nullptr == dest || nullptr == extMem) {
    QNN_ERROR("Received nullptr");
    return false;
  }

  Qnn_ClientBuffer_t clientBuffer;
  clientBuffer.data     = extMem;
  clientBuffer.dataSize = QNN_TENSOR_GET_CLIENT_BUF(dest).dataSize;
  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSORMEMTYPE_RAW);
  QNN_TENSOR_SET_CLIENT_BUF(dest, clientBuffer);
  m_extBufferTensors.insert(dest);
  return true;
}
