// Compile guard for the D3D12 hook diagnostic contract.
//
// This intentionally does not link d3d12_hooks.cpp or touch Detours/D3D12
// runtime state. It catches accidental removal/renaming of the fields used by
// smoke tooling and native logs to verify hook telemetry.
#include "../src/d3d12_hooks.h"

#include <cassert>

int main() {
    oss_gaussian::D3D12HookStatus status{};

    status.state = oss_gaussian::D3D12HookInstallState::Degraded;
    status.installed = true;
    status.degraded = true;
    status.present_hooked = true;
    status.present1_hooked = true;
    status.execute_command_lists_hooked = true;
    status.create_device_hooked = true;
    status.create_command_queue_hooked = true;
    status.swap_chain_creation_hooked = true;
    status.readback_degraded = false;
    status.last_detour_error = 0;
    status.last_probe_hresult = 0;
    status.frame_count = 1;
    status.present_count = 2;
    status.present1_count = 3;
    status.create_device_count = 4;
    status.create_command_queue_count = 5;
    status.execute_command_lists_count = 6;
    status.swap_chain_creation_count = 7;
    status.capture_keep_count = 8;
    status.degraded_capture_count = 9;

    assert(status.state == oss_gaussian::D3D12HookInstallState::Degraded);
    assert(status.execute_command_lists_hooked);
    assert(status.create_device_hooked);
    assert(status.create_command_queue_hooked);
    assert(status.execute_command_lists_count == 6);
    assert(status.swap_chain_creation_count == 7);
    return 0;
}
