#include <mitsuba/render/torch_bridge/torch_bridge.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

#include <mutex>

struct FM_BRIDGE_API ModelContainer {
    torch::jit::script::Module model;
    bool loaded = false;
    torch::Device device = torch::kCUDA;

    void LoadModel(const char* path, bool use_gpu) {
        // std::cout<<"load model"<<std::endl;
        
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
            std::cerr << "[ModelContainer] Model moved to CUDA. "
                        << "Device count = " << torch::cuda::device_count() << std::endl;
        } else {
            if (!torch::cuda::is_available()) {
                std::cerr << "[ModelContainer] WARNING: GPU requested but "
                            << "torch::cuda::is_available() returned false! "
                            << "Falling back to CPU." << std::endl;
            }
            device = torch::kCPU;
        }
    }

};

// heun3 flow matching ode
torch::Tensor exact_divergence(const torch::Tensor& ut, const torch::Tensor& xt) {
    auto div = torch::zeros({xt.size(0)}, xt.options());
    for (int64_t i = 0; i < ut.size(1); ++i) {
    auto grad_outputs = torch::ones_like(ut.select(1, i));
    auto dui_dx = torch::autograd::grad(
        {ut.select(1, i)}, {xt}, {grad_outputs}, /*retain_graph=*/true)[0];
    div = div + dui_dx.select(1, i);
    }
    return div;
}

std::pair<torch::Tensor, torch::Tensor> heun3_step(
    ModelContainer& impl, const torch::Tensor& t0, const torch::Tensor& t1,
    const torch::Tensor& x0, const torch::Tensor& points, const torch::Tensor& logdet0) {
    auto dt = t1 - t0;
    int64_t batch_size = x0.size(0);

    torch::Tensor u1, div1, u3, div3;
    {
        torch::AutoGradMode enable(true);
        auto xt = x0.clone().requires_grad_();
        auto t0_batched = t0.expand({batch_size});
        auto ut = impl.model.forward({xt, points, t0_batched}).toTensor();
        auto div = exact_divergence(ut, xt);
        u1 = ut.detach(); div1 = div.detach();
    }

    torch::Tensor u2;
    {
        torch::NoGradGuard no_grad;
        auto x2 = x0 + (dt * u1) * (1.0 / 3.0);          // grouping matters, see PART 1b
        auto t_batched = (t0 + dt * (1.0 / 3.0)).expand({batch_size});
        u2 = impl.model.forward({x2, points, t_batched}).toTensor();
    }

    auto x3 = x0 + dt * (u2 * (2.0 / 3.0));              // grouping matters, see PART 1b
    {
        torch::AutoGradMode enable(true);
        auto xt = x3.clone().requires_grad_();
        auto t_batched = (t0 + dt * (2.0 / 3.0)).expand({batch_size});
        auto ut = impl.model.forward({xt, points, t_batched}).toTensor();
        auto div = exact_divergence(ut, xt);
        u3 = ut.detach(); div3 = div.detach();
    }

    auto x1 = x0 + dt * (u1 * (1.0 / 4.0) + u3 * (3.0 / 4.0));         // grouping matters
    auto logdet1 = logdet0 + dt * (div1 * (1.0 / 4.0) + div3 * (3.0 / 4.0)); // grouping matters
    return {x1, logdet1};
}

// --- sample_uniform_disk ---
torch::Tensor sample_uniform_disk(int64_t n, const torch::TensorOptions& options) {
    auto disk_r = torch::sqrt(torch::rand({n}, options));
    auto disk_theta = torch::rand({n}, options) * 2.0 * M_PI;
    return torch::stack({disk_r * torch::cos(disk_theta), disk_r * torch::sin(disk_theta)}, -1);
}

// --- inv_symlog ---
torch::Tensor inv_symlog(const torch::Tensor& y) {
    return torch::sign(y) * torch::expm1(torch::abs(y));
}

// --- inv_symlog_logdet ---
torch::Tensor inv_symlog_logdet(const torch::Tensor& model_output) {
    return model_output.abs().sum(-1);
}

// --- inv_stereographic_logdet ---
torch::Tensor inv_stereographic_logdet(const torch::Tensor& uv) {
    auto r2 = (uv * uv).sum(-1);
    return std::log(4.0) - 2.0 * torch::log1p(r2);
}

// --- stereographic_to_direction ---
torch::Tensor stereographic_to_direction(const torch::Tensor& uv_in, bool use_inv_symlog = true) {
    auto uv = use_inv_symlog ? inv_symlog(uv_in) : uv_in;
    auto u = uv.select(1, 0), v = uv.select(1, 1);
    auto denom = 1.0 + u * u + v * v;
    auto x = 2.0 * u / denom;
    auto y = 2.0 * v / denom;
    auto z = (denom - 2.0) / denom;
    auto d = torch::stack({x, y, z}, 1);
    return d / d.norm(2, 1, /*keepdim=*/true);
}

std::pair<torch::Tensor, torch::Tensor> sample_and_likelihood(
    ModelContainer& impl, const torch::Tensor& x_0,
    const torch::Tensor& points,
    const std::function<torch::Tensor(const torch::Tensor&)>& log_p0,
    const torch::Tensor& time_grid) {
    torch::NoGradGuard no_grad;
    auto x = x_0;
    auto logdet = torch::zeros({x_0.size(0)}, x_0.options());
    for (int64_t i = 0; i + 1 < time_grid.size(0); ++i) {
        auto t0 = time_grid[i];
        auto t1 = time_grid[i + 1];
        std::tie(x, logdet) = heun3_step(impl, t0, t1, x, points, logdet);
    }
    return {x, log_p0(x_0) - logdet};
}

// --- sample_flow_direction ---
std::pair<torch::Tensor, torch::Tensor> sample_flow_direction_alignment(
        ModelContainer& impl, const torch::Tensor& points,
        const torch::Tensor& x_0,
        const torch::Tensor& obj_to_world_rotation,
        const torch::Tensor& time_grid,
        const torch::TensorOptions& options) {
    int64_t n = points.size(0);
    double disk_log_density = -std::log(M_PI);  // uniform density on the unit disk (area = pi)

    auto log_p0 = [&](const torch::Tensor& x) {
        return torch::full({x.size(0)}, disk_log_density, x.options());
    };

    auto [model_output, log_det] = sample_and_likelihood(impl, x_0, points, log_p0, time_grid);

   auto sampled_direction = torch::matmul(stereographic_to_direction(model_output), obj_to_world_rotation.t());
    sampled_direction = sampled_direction / sampled_direction.norm(2, -1, /*keepdim=*/true);

    auto inv_symlog_output = inv_symlog(model_output);
    auto transformation_logdet = inv_symlog_logdet(model_output) + inv_stereographic_logdet(inv_symlog_output);
    auto local_log_p = log_det - transformation_logdet;
    // 1.0 / exp(x), not exp(-x) - see PART 1b and sample_flow_direction's own comment.
    auto sampled_weight = 1.0 / torch::exp(local_log_p);

    return {sampled_direction, sampled_weight};
}

std::pair<torch::Tensor, torch::Tensor> sample_flow_direction(
        ModelContainer& impl, const torch::Tensor& points,
        const torch::Tensor& obj_to_world_rotation,
        const torch::Tensor& time_grid,
        const torch::TensorOptions& options) {
    int64_t n = points.size(0);
    double disk_log_density = -std::log(M_PI);  // uniform density on the unit disk (area = pi)

    auto log_p0 = [&](const torch::Tensor& x) {
        return torch::full({x.size(0)}, disk_log_density, x.options());
    };

    auto x_0 = sample_uniform_disk(n, options);
    auto [model_output, log_det] = sample_and_likelihood(impl, x_0, points, log_p0, time_grid);

   auto sampled_direction = torch::matmul(stereographic_to_direction(model_output), obj_to_world_rotation.t());
    sampled_direction = sampled_direction / sampled_direction.norm(2, -1, /*keepdim=*/true);

    auto inv_symlog_output = inv_symlog(model_output);
    auto transformation_logdet = inv_symlog_logdet(model_output) + inv_stereographic_logdet(inv_symlog_output);
    auto local_log_p = log_det - transformation_logdet;
    // 1.0 / exp(x), not exp(-x) - see PART 1b and sample_flow_direction's own comment.
    auto sampled_weight = 1.0 / torch::exp(local_log_p);

    return {sampled_direction, sampled_weight};
}

VisnetModelBridge& VisnetModelBridge::instance(const char* path, bool use_gpu) {
    static VisnetModelBridge inst(path, use_gpu);   // turn 155 的 Meyer's Singleton,寫法完全一致
    return inst;
}

VisnetModelBridge::VisnetModelBridge(const char* path, bool use_gpu) {
    // torch::set_num_threads(1);     
    // std::cout<<"constructor"<<std::endl;
    m_impl = new ModelContainer();
    m_impl->LoadModel(path, use_gpu);
}

void VisnetModelBridge::vis_forward(const float* input_data, int* output, int data_size, float threshold) {
    if (!m_impl->loaded) {
        throw std::runtime_error("VisnetModelBridge: model not loaded");
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

VisnetModelBridge::~VisnetModelBridge() { delete m_impl; }

FlowModelBridge& FlowModelBridge::instance(const char* path, bool use_gpu) {
    static FlowModelBridge inst(path, use_gpu);   // turn 155 的 Meyer's Singleton,寫法完全一致
    return inst;
}

FlowModelBridge::FlowModelBridge(const char* path, bool use_gpu) {
    // torch::set_num_threads(1);     
    // std::cout<<"constructor"<<std::endl;
    m_impl = new ModelContainer();
    m_impl->LoadModel(path, use_gpu);
}

void FlowModelBridge::flow_forward(const float* input_data, float* output_dir, float* output_weights, int data_size, int step_count) {
    auto option = torch::TensorOptions().dtype(torch::kFloat32).device(m_impl->device);
    auto points_tensor = torch::from_blob((void*)input_data, {data_size, 6}, torch::kFloat32).to(m_impl->device, /*non_blocking=*/false);

    int p = 2;
    auto time_grid = torch::linspace(0.0, 1.0, step_count + 1, option);

    // TODO: connect to sample_flow_direction

}

void FlowModelBridge::test_flow_forward() {
    if (!m_impl->loaded) {
        throw std::runtime_error("FlowModelBridge: model not loaded");
    }

    int n = 4;
    std::vector<float> c = {
        0.100000001f, 0.200000003f, 0.300000012f, -0.100000001f, 0.0500000007f, 0.200000003f, 0.0f, 0.0f, 0.5f, 0.200000003f, -0.300000012f, 0.100000001f, -0.200000003f, 0.100000001f, 0.0f, 0.0f, 0.400000006f, -0.100000001f, 0.300000012f, -0.200000003f, 0.100000001f, 0.200000003f, 0.200000003f, 0.300000012f
    };

    std::vector<float> x_0 ={
        0.0f, 0.0f, 0.5f, 0.0f, 0.0f, -0.5f, 0.300000012f, 0.400000006f
    };

    auto c_tensor = torch::from_blob((void*)c.data(), {n, 6}, torch::kFloat32).to(m_impl->device, /*non_blocking=*/false);
    auto x_0_tensor = torch::from_blob((void*)x_0.data(), {n, 2}, torch::kFloat32).to(m_impl->device, /*non_blocking=*/false);

    int n_step = 10;
    int p = 2;
    auto option = torch::TensorOptions().dtype(torch::kFloat32).device(m_impl->device);
    auto time_grid = torch::linspace(0.0, 1.0, n_step + 1, option);
    // non-uniform time steps, biased towards t=1
    time_grid = 1.0 - torch::pow(1.0 - time_grid, p);

    torch::Tensor identity = torch::eye(3, option);

    auto [direction, weight] = sample_flow_direction_alignment(*m_impl, c_tensor, x_0_tensor, identity, time_grid, option);

    std::cout<<"direction: "<<direction<<std::endl;
    std::cout<<"weight: "<<weight<<std::endl;
}

FlowModelBridge::~FlowModelBridge() { delete m_impl; }


extern "C" {

FM_BRIDGE_API void torch_load_visnet_model(const char* path, bool use_gpu){
    auto &w = VisnetModelBridge::instance(path, use_gpu);
}

FM_BRIDGE_API void torch_visnet_forward(float* input_data, int* output, int data_size, float threshold){
    auto &w = VisnetModelBridge::instance(nullptr); // 使用已經載入的模型
    w.vis_forward(input_data, output, data_size, threshold);
}

FM_BRIDGE_API void torch_load_flow_model(const char* path, bool use_gpu){
    auto &w = FlowModelBridge::instance(path, use_gpu);
}

FM_BRIDGE_API void torch_flow_forward(float* input_data, float* output_dir, float* output_weights, int data_size, int step_count){
    auto &w = FlowModelBridge::instance(nullptr); // 使用已經載入的模型
    w.flow_forward(input_data, output_dir, output_weights, data_size, step_count);
}

FM_BRIDGE_API void torch_test_flow_forward(){
    auto &w = FlowModelBridge::instance(nullptr);
    w.test_flow_forward();
    
}

} // extern "C"
