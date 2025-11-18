//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>

#include "qualla/env.hpp"

namespace fs = std::filesystem;

namespace qualla {

static uint64_t s_nameCounter{0};

Env::Env(const json& conf) {
  _path.models = fs::path();
  _path.cache  = fs::path();

  if (conf.contains("path")) {
    const json& p = conf["path"];

    if (p.contains("models"))
      _path.models = fs::path(p["models"].get<std::string>()).make_preferred();
    if (p.contains("cache")) _path.cache = fs::path(p["cache"].get<std::string>()).make_preferred();
  }
  _name = "env" + std::to_string(s_nameCounter++);
}

Env::~Env() {}

bool Env::update(std::shared_ptr<Env> env) {
  if (_name == env->getName()) return true;
  _name     = env->getName();
  m_loggers = env->getLogger();
  _path     = env->getPath();
  return true;
}

std::shared_ptr<Env> Env::create(const qualla::json& conf) { return std::make_shared<Env>(conf); }

std::shared_ptr<Env> Env::create(std::istream& json_stream) {
  return create(json::parse(json_stream));
}

std::shared_ptr<Env> Env::create(const std::string& json_str) {
  return create(json::parse(json_str));
}

}  // namespace qualla
