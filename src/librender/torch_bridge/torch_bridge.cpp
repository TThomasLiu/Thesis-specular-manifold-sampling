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

FlowModelBridge& FlowModelBridge::instance(const char* path) {
    static FlowModelBridge inst(path);   // turn 155 的 Meyer's Singleton,寫法完全一致
    return inst;
}

FlowModelBridge::FlowModelBridge(const char* path) {
    // torch::set_num_threads(1);     
    std::cout<<"constructor"<<std::endl;
    m_impl = new Impl();
    m_impl->model = torch::jit::load(path);
    m_impl->model.to(torch::kCPU);
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
    m_impl->model.eval();
}

void FlowModelBridge::step(const float* xt, const float* c,
                            float t_start, float t_end,
                            float* out_xt) {
    if (!m_impl->loaded) {
        throw std::runtime_error("FlowModelBridge: model not loaded");
    }
    torch::NoGradGuard no_grad;

    auto c_tensor   = torch::from_blob((void*)c,   {1, 6}, torch::kFloat32).clone();
    auto logit = m_impl->model.get_method("forward")({c_tensor}).toTensor();

    // 如果訓練時用 BCEWithLogitsLoss,forward 輸出是 raw logit,需要自己套 sigmoid
    auto prob = torch::sigmoid(logit);

    // 依照你們討論過的 threshold trade-off(FN 代價高於 FP,建議 < 0.5)
    constexpr float kThreshold = 0.5f;  // 待用 PR curve 決定實際值
    bool is_visible = prob.item<float>() > kThreshold;
}

void FlowModelBridge::vis_forward(const float* input_data, int* output, int data_size) {
    if (!m_impl->loaded) {
        throw std::runtime_error("FlowModelBridge: model not loaded");
    }
    torch::NoGradGuard no_grad;

    auto c_tensor   = torch::from_blob((void*)input_data,   {data_size, 6}, torch::kFloat32).to(m_impl->device, false);
    auto logit = m_impl->model.get_method("forward")({c_tensor}).toTensor();

    // 如果訓練時用 BCEWithLogitsLoss,forward 輸出是 raw logit,需要自己套 sigmoid
    auto prob = torch::sigmoid(logit).squeeze(-1);

    // 依照你們討論過的 threshold trade-off(FN 代價高於 FP,建議 < 0.5)
    constexpr float kThreshold = 0.5f;  // 待用 PR curve 決定實際值
    auto pred = (prob > kThreshold).to(torch::kInt32).contiguous().to(torch::kCPU);

    // 把結果寫回呼叫端提供的 output buffer
    std::memcpy(output, pred.data_ptr<int32_t>(), data_size * sizeof(int32_t));
}

FlowModelBridge::~FlowModelBridge() { delete m_impl; }

extern "C" {

FM_BRIDGE_API void torch_sanity_check(void) {
    // try {
    //     torch::Tensor t = torch::rand({2, 3});
    //     std::cout << "[fm_torch_sanity_check] tensor:\n" << t << std::endl;
    //     std::cout << "[fm_torch_sanity_check] sum = " << t.sum().item<float>() << std::endl;
    //     std::cout << "[fm_torch_sanity_check] CUDA available = "
    //               << (torch::cuda::is_available() ? "yes" : "no") << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "[fm_torch_sanity_check] EXCEPTION: " << e.what() << std::endl;
    // }
}


FM_BRIDGE_API void torch_load_model(const char* path){
    auto &w = FlowModelBridge::instance(path);
    // std::cout << "[fm_torch_load_model] model container " << &w << std::endl;
}


FM_BRIDGE_API void torch_test_model(const char* path){
    auto &w = FlowModelBridge::instance(path);
    float xt[2] = {0.1f, -0.2f};
    float c[6]   = {0.5f, 0.3f, -0.1f,   // x_s
                     0.0f, 0.2f,  0.4f}; // x_l
    float t_start = 0.0f, t_end = 0.5f;
    float out[2];

    w.step(xt, c, t_start, t_end, out);
    // std::cout << std::fixed << std::setprecision(8);
    // std::cout << "[torch_test_flow_step] output = ("
    //         << out[0] << ", " << out[1] << ")" << std::endl;

}

FM_BRIDGE_API void torch_test_vismodel(float* input_data, int* output, int data_size){
    auto &w = FlowModelBridge::instance(nullptr); // 使用已經載入的模型
    w.vis_forward(input_data, output, data_size);
}

} // extern "C"
