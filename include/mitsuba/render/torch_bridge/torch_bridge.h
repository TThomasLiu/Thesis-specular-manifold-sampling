#pragma once
#include <string>
#if defined(_WIN32)
  #define FM_BRIDGE_API __declspec(dllexport)
#else
  #define FM_BRIDGE_API __attribute__((visibility("default")))
#endif

// 用純 C++ 陣列/vector 傳資料,完全不暴露 torch::Tensor 這個型別給外部
struct FM_BRIDGE_API ModelContainer;


class FM_BRIDGE_API VisnetModelBridge {
public:
    static VisnetModelBridge& instance(const char* path, bool use_gpu = false);

    void vis_forward(const float* input_data, int* output, int data_size, float threshold);
        ~VisnetModelBridge();
private:
    VisnetModelBridge(const char* path, bool use_gpu);
    ModelContainer* m_impl;
};

class FM_BRIDGE_API FlowModelBridge {
public:
    static FlowModelBridge& instance(const char* path, bool use_gpu = false);

    void flow_forward(const float* input_data, float* output_dir, float* output_weights, int data_size, int step_count);
    void test_flow_forward();
        ~FlowModelBridge();
private:
    FlowModelBridge(const char* path, bool use_gpu);
    ModelContainer* m_impl;
};


#ifdef __cplusplus
extern "C" {
#endif

void torch_load_visnet_model(const char* path, bool use_gpu = false);
void torch_visnet_forward(float* input_data, int* output, int data_size, float threshold = 0.4f);
void torch_load_flow_model(const char* path, bool use_gpu = false);
void torch_flow_forward(float* input_data, float* output_dir, float* output_weights, int data_size, int step_count = 10);
void torch_test_flow_forward();

#ifdef __cplusplus
}
#endif