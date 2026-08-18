#include <mitsuba/render/torch_bridge/torch_bridge.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

#include <mutex>

struct FlowModelBridge::Impl {
    torch::jit::script::Module model;
    bool loaded = false;
    torch::Device device = torch::kCUDA;

    void LoadModel(const char* path, bool use_gpu) {
        std::cout<<"load model"<<std::endl;
        
        model = torch::jit::load(path);
        model.eval();

        if (use_gpu){
            model.to(torch::kCUDA);
            device = torch::kCUDA;
        }else{
            model.to(torch::kCPU);
            device = torch::kCPU;
        }

        loaded = true;

        if (device == torch::kCUDA && torch::cuda::is_available()) {
            model.to(torch::kCUDA);
            device = torch::kCUDA;
            std::cerr << "[FlowModelBridge] Model moved to CUDA. "
                        << "Device count = " << torch::cuda::device_count() << std::endl;
        } else {
            if (!torch::cuda::is_available()) {
                std::cerr << "[FlowModelBridge] WARNING: GPU requested but "
                            << "torch::cuda::is_available() returned false! "
                            << "Falling back to CPU." << std::endl;
            }
            device = torch::kCPU;
        }
    }

};

FlowModelBridge& FlowModelBridge::instance(const char* path, bool use_gpu) {
    static FlowModelBridge inst(path, use_gpu);   // turn 155 的 Meyer's Singleton,寫法完全一致
    return inst;
}

FlowModelBridge::FlowModelBridge(const char* path, bool use_gpu) {
    // torch::set_num_threads(1);     
    std::cout<<"constructor"<<std::endl;
    m_impl = new Impl();
    m_impl->LoadModel(path, use_gpu);
}

void FlowModelBridge::vis_forward(const float* input_data, int* output, int data_size, float threshold) {
    if (!m_impl->loaded) {
        throw std::runtime_error("FlowModelBridge: model not loaded");
    }
    torch::NoGradGuard no_grad;

    auto c_tensor   = torch::from_blob((void*)input_data,   {data_size, 6}, torch::kFloat32).to(m_impl->device, /*non_blocking=*/false);
    auto logit = m_impl->model.forward({c_tensor}).toTensor();
    
    float kThreshold = threshold; 
    
    const float kLogitThreshold = std::log(kThreshold / (1.0f - kThreshold));
    auto mask = (logit.squeeze(-1) > kLogitThreshold).to(torch::kInt32);
    auto out_tensor = torch::from_blob(output, {data_size}, torch::kInt32);
    out_tensor.copy_(mask);
}

FlowModelBridge::~FlowModelBridge() { delete m_impl; }

extern "C" {

FM_BRIDGE_API void torch_load_model(const char* path, bool use_gpu){
    auto &w = FlowModelBridge::instance(path, use_gpu);
}

FM_BRIDGE_API void torch_test_vismodel(float* input_data, int* output, int data_size, float threshold){
    auto &w = FlowModelBridge::instance(nullptr); // 使用已經載入的模型
    w.vis_forward(input_data, output, data_size, threshold);
}

} // extern "C"
