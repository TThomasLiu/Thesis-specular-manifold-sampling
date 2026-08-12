#include <mitsuba/render/torch_bridge/torch_bridge.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

#include <mutex>

struct FlowModelBridge::Impl {
    torch::jit::script::Module model;
    bool loaded = false;
    torch::Device device = torch::kCUDA;
};

FlowModelBridge& FlowModelBridge::instance(const char* path, bool use_gpu) {
    static FlowModelBridge inst(path, use_gpu);   // turn 155 的 Meyer's Singleton,寫法完全一致
    return inst;
}

FlowModelBridge::FlowModelBridge(const char* path, bool use_gpu) {
    // torch::set_num_threads(1);     
    std::cout<<"constructor"<<std::endl;
    m_impl = new Impl();
    m_impl->model = torch::jit::load(path);
    m_impl->model.eval();

    if (use_gpu){
        m_impl->model.to(torch::kCUDA);
        m_impl->device = torch::kCUDA;
    }else{
        m_impl->model.to(torch::kCPU);
        m_impl->device = torch::kCPU;
    }

    m_impl->loaded = true;

    if (m_impl->device == torch::kCUDA && torch::cuda::is_available()) {
        m_impl->model.to(torch::kCUDA);
        m_impl->device = torch::kCUDA;
        std::cerr << "[FlowModelBridge] Model moved to CUDA. "
                    << "Device count = " << torch::cuda::device_count() << std::endl;
    } else {
        if (!torch::cuda::is_available()) {
            std::cerr << "[FlowModelBridge] WARNING: GPU requested but "
                        << "torch::cuda::is_available() returned false! "
                        << "Falling back to CPU." << std::endl;
        }
        m_impl->device = torch::kCPU;
    }
    // m_impl->model.to(torch::kHalf);
    // m_impl->model = torch::jit::freeze(m_impl->model);            // 2. 再 freeze
    // m_impl->model = torch::jit::optimize_for_inference(m_impl->model);  // 3. 最後優化
}

void FlowModelBridge::vis_forward(const float* input_data, int* output, int data_size) {
    if (!m_impl->loaded) {
        throw std::runtime_error("FlowModelBridge: model not loaded");
    }
    torch::NoGradGuard no_grad;

    // auto c_tensor   = torch::from_blob((void*)input_data,   {data_size, 6}, torch::kFloat32).to(m_impl->device, torch::kHalf, /*non_blocking=*/false);;
    auto c_tensor   = torch::from_blob((void*)input_data,   {data_size, 6}, torch::kFloat32).to(m_impl->device, /*non_blocking=*/false);;
    auto logit = m_impl->model.get_method("forward")({c_tensor}).toTensor();

    // 如果訓練時用 BCEWithLogitsLoss,forward 輸出是 raw logit,需要自己套 sigmoid
    
    constexpr float kThreshold = 0.4f; 
    // auto prob = torch::sigmoid(logit).squeeze(-1);
    // auto pred = (prob > kThreshold).to(torch::kInt32).contiguous().to(torch::kCPU);
    
    const float kLogitThreshold = std::log(kThreshold / (1.0f - kThreshold)); 
    auto pred = (logit.squeeze(-1) > kLogitThreshold).to(torch::kInt32).contiguous().to(torch::kCPU);


    // 把結果寫回呼叫端提供的 output buffer
    std::memcpy(output, pred.data_ptr<int32_t>(), data_size * sizeof(int32_t));
}

FlowModelBridge::~FlowModelBridge() { delete m_impl; }

extern "C" {

FM_BRIDGE_API void torch_load_model(const char* path, bool use_gpu){
    auto &w = FlowModelBridge::instance(path, use_gpu);
}

FM_BRIDGE_API void torch_test_vismodel(float* input_data, int* output, int data_size){
    auto &w = FlowModelBridge::instance(nullptr); // 使用已經載入的模型
    w.vis_forward(input_data, output, data_size);
}

} // extern "C"
