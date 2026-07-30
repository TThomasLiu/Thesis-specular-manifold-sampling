#include <mitsuba/render/torch_bridge/torch_bridge.h>
#include <torch/torch.h>
#include <iostream>

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

} // extern "C"