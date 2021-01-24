#pragma once
#include <random>
#include <chrono>
#include <thread>



using GeneratorType = std::default_random_engine;

class Generator {
  GeneratorType generator;
  Generator() { 
    uint64_t seed = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
    seed += (uint64_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
    generator = GeneratorType(seed);
  }
public:
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;
  static GeneratorType& Get() {
    static thread_local Generator gen;
    return gen.generator;
  }
};

