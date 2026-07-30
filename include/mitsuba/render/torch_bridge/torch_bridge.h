#pragma once
#if defined(_WIN32)
  #define FM_BRIDGE_API __declspec(dllexport)
#else
  #define FM_BRIDGE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

void torch_sanity_check(void);

#ifdef __cplusplus
}
#endif