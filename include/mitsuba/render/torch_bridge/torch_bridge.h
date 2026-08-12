#pragma once
#include <string>
#if defined(_WIN32)
  #define FM_BRIDGE_API __declspec(dllexport)
#else
  #define FM_BRIDGE_API __attribute__((visibility("default")))
#endif

// 用純 C++ 陣列/vector 傳資料,完全不暴露 torch::Tensor 這個型別給外部
class FM_BRIDGE_API FlowModelBridge {
public:
    static FlowModelBridge& instance(const char* path);
    
    void vis_forward(const float* input_data, int* output, int data_size);
        ~FlowModelBridge();
private:
    FlowModelBridge(const char* path);
    struct Impl;              // Pimpl:把 torch::jit::script::Module 藏在這裡
    Impl* m_impl;
};

#ifdef __cplusplus
extern "C" {
#endif

void torch_load_model(const char* path);
void torch_test_vismodel(float* input_data, int* output, int data_size);

#ifdef __cplusplus
}
#endif