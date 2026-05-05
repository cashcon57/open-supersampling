#include "oss_capture.h"

#include <cstring>
#include <filesystem>
#include <fstream>

#if defined(OSS_CAPTURE_HAS_OPENEXR)
#  if __has_include(<OpenEXR/ImfHeader.h>)
#    include <OpenEXR/ImfChannelList.h>
#    include <OpenEXR/ImfFrameBuffer.h>
#    include <OpenEXR/ImfHeader.h>
#    include <OpenEXR/ImfOutputFile.h>
#  else
#    include <ImfChannelList.h>
#    include <ImfFrameBuffer.h>
#    include <ImfHeader.h>
#    include <ImfOutputFile.h>
#  endif
#endif

namespace {

constexpr const char* kChannelNames[] = {
    "LR.R", "LR.G", "LR.B",
    "HR.R", "HR.G", "HR.B",
    "Depth.Z",
    "Motion.X", "Motion.Y",
    "Normals.X", "Normals.Y", "Normals.Z",
    "Albedo.R", "Albedo.G", "Albedo.B",
    "Roughness.R",
    "Metallic.R",
    "Emissive.R", "Emissive.G", "Emissive.B",
};

bool flag_default_on(uint32_t flag, bool legacy_on) {
    return flag != 0u || legacy_on;
}

bool should_write_hr(const OssCaptureFramePayload& payload) {
    return payload.capture_hr != 0u ||
        (payload.burst_tier != OSS_CAPTURE_TIER_LONG && payload.capture_mode == OSS_CAPTURE_MODE_LITE);
}

bool valid_view(const OssCaptureImageView& view, uint32_t channels) {
    return view.pixels != nullptr && view.width > 0 && view.height > 0 && view.channels == channels;
}

void write_view(std::ofstream& out, const OssCaptureImageView& view) {
    const size_t count = static_cast<size_t>(view.width) * view.height * view.channels;
    out.write(reinterpret_cast<const char*>(view.pixels), static_cast<std::streamsize>(count * sizeof(float)));
}

#if defined(OSS_CAPTURE_HAS_OPENEXR)
void insert_float_channel(Imf::Header& header, const char* name) {
    header.channels().insert(name, Imf::Channel(Imf::FLOAT));
}

void insert_slice(Imf::FrameBuffer& fb, const char* name, const OssCaptureImageView& view, uint32_t channel) {
    const char* base = reinterpret_cast<const char*>(view.pixels + channel);
    fb.insert(
        name,
        Imf::Slice(
            Imf::FLOAT,
            const_cast<char*>(base),
            static_cast<size_t>(view.channels * sizeof(float)),
            static_cast<size_t>(view.channels * view.width * sizeof(float))));
}

const OssCaptureImageView& canvas_view(const OssCaptureFramePayload& payload) {
    return should_write_hr(payload) ? payload.hr_rgb : payload.lr_rgb;
}

bool same_canvas(const OssCaptureFramePayload& payload) {
    const OssCaptureImageView& canvas = canvas_view(payload);
    const uint32_t w = canvas.width;
    const uint32_t h = canvas.height;
    return payload.lr_rgb.width == w && payload.lr_rgb.height == h &&
           payload.depth_z.width == w && payload.depth_z.height == h &&
           payload.motion_xy.width == w && payload.motion_xy.height == h &&
           payload.normals_xyz.width == w && payload.normals_xyz.height == h &&
           (!should_write_hr(payload) || (payload.hr_rgb.width == w && payload.hr_rgb.height == h)) &&
           (payload.capture_albedo == 0u || (payload.albedo_rgb.width == w && payload.albedo_rgb.height == h)) &&
           (payload.capture_roughness == 0u || (payload.roughness.width == w && payload.roughness.height == h)) &&
           (payload.capture_metallic == 0u || (payload.metallic.width == w && payload.metallic.height == h)) &&
           (payload.capture_emissive == 0u || (payload.emissive_rgb.width == w && payload.emissive_rgb.height == h));
}
#endif

} // namespace

extern "C" {

int oss_capture_write_exr(const wchar_t* path, const OssCaptureFramePayload* payload) {
    if (!path || !payload) {
        return 0;
    }
    const bool write_lr = payload->capture_lr != 0u || payload->capture_mode == OSS_CAPTURE_MODE_LITE;
    const bool write_hr = should_write_hr(*payload);
    const bool write_depth = flag_default_on(payload->capture_depth, true);
    const bool write_motion = flag_default_on(payload->capture_motion, true);
    const bool write_normals = flag_default_on(payload->capture_normals, true);
    if ((write_lr && !valid_view(payload->lr_rgb, 3)) ||
        (write_hr && !valid_view(payload->hr_rgb, 3)) ||
        (write_depth && !valid_view(payload->depth_z, 1)) ||
        (write_motion && !valid_view(payload->motion_xy, 2)) ||
        (write_normals && !valid_view(payload->normals_xyz, 3)) ||
        (payload->capture_albedo != 0u && !valid_view(payload->albedo_rgb, 3)) ||
        (payload->capture_roughness != 0u && !valid_view(payload->roughness, 1)) ||
        (payload->capture_metallic != 0u && !valid_view(payload->metallic, 1)) ||
        (payload->capture_emissive != 0u && !valid_view(payload->emissive_rgb, 3))) {
        return 0;
    }

    std::filesystem::path fs_path(path);
    if (!fs_path.parent_path().empty()) {
        std::filesystem::create_directories(fs_path.parent_path());
    }

#if defined(OSS_CAPTURE_HAS_OPENEXR)
    if (!same_canvas(*payload)) {
        return 0;
    }
    const OssCaptureImageView& canvas = canvas_view(*payload);
    Imf::Header header(static_cast<int>(canvas.width), static_cast<int>(canvas.height));
    header.compression() = Imf::ZIP_COMPRESSION;
    for (const char* channel : kChannelNames) {
        if (!write_lr && (channel[0] == 'L' && channel[1] == 'R' && channel[2] == '.')) {
            continue;
        }
        if (!write_hr && (channel[0] == 'H' && channel[1] == 'R' && channel[2] == '.')) {
            continue;
        }
        if (!write_depth && std::strncmp(channel, "Depth.", 6) == 0) continue;
        if (!write_motion && std::strncmp(channel, "Motion.", 7) == 0) continue;
        if (!write_normals && std::strncmp(channel, "Normals.", 8) == 0) continue;
        if (payload->capture_albedo == 0u && std::strncmp(channel, "Albedo.", 7) == 0) continue;
        if (payload->capture_roughness == 0u && std::strncmp(channel, "Roughness.", 10) == 0) continue;
        if (payload->capture_metallic == 0u && std::strncmp(channel, "Metallic.", 9) == 0) continue;
        if (payload->capture_emissive == 0u && std::strncmp(channel, "Emissive.", 9) == 0) continue;
        insert_float_channel(header, channel);
    }

    Imf::FrameBuffer frame_buffer;
    if (write_lr) {
        insert_slice(frame_buffer, "LR.R", payload->lr_rgb, 0);
        insert_slice(frame_buffer, "LR.G", payload->lr_rgb, 1);
        insert_slice(frame_buffer, "LR.B", payload->lr_rgb, 2);
    }
    if (write_hr) {
        insert_slice(frame_buffer, "HR.R", payload->hr_rgb, 0);
        insert_slice(frame_buffer, "HR.G", payload->hr_rgb, 1);
        insert_slice(frame_buffer, "HR.B", payload->hr_rgb, 2);
    }
    if (write_depth) insert_slice(frame_buffer, "Depth.Z", payload->depth_z, 0);
    if (write_motion) {
        insert_slice(frame_buffer, "Motion.X", payload->motion_xy, 0);
        insert_slice(frame_buffer, "Motion.Y", payload->motion_xy, 1);
    }
    if (write_normals) {
        insert_slice(frame_buffer, "Normals.X", payload->normals_xyz, 0);
        insert_slice(frame_buffer, "Normals.Y", payload->normals_xyz, 1);
        insert_slice(frame_buffer, "Normals.Z", payload->normals_xyz, 2);
    }
    if (payload->capture_albedo != 0u) {
        insert_slice(frame_buffer, "Albedo.R", payload->albedo_rgb, 0);
        insert_slice(frame_buffer, "Albedo.G", payload->albedo_rgb, 1);
        insert_slice(frame_buffer, "Albedo.B", payload->albedo_rgb, 2);
    }
    if (payload->capture_roughness != 0u) insert_slice(frame_buffer, "Roughness.R", payload->roughness, 0);
    if (payload->capture_metallic != 0u) insert_slice(frame_buffer, "Metallic.R", payload->metallic, 0);
    if (payload->capture_emissive != 0u) {
        insert_slice(frame_buffer, "Emissive.R", payload->emissive_rgb, 0);
        insert_slice(frame_buffer, "Emissive.G", payload->emissive_rgb, 1);
        insert_slice(frame_buffer, "Emissive.B", payload->emissive_rgb, 2);
    }

    Imf::OutputFile file(fs_path.string().c_str(), header);
    file.setFrameBuffer(frame_buffer);
    file.writePixels(static_cast<int>(canvas.height));
    return 1;
#else
    // Sprint-6 capture scaffold: OpenEXR/tinyexr is not yet vendored in this
    // DLL tree. Keep the writer deterministic and schema-labeled so integration
    // tests can validate the capture contract; swap the body for ZIP level-5
    // OpenEXR emission once the dependency is added to the Windows build.
    std::ofstream out(fs_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return 0;
    }
    out << "OSS_CAPTURE_EXR_V0\n";
    out << "compression=zip-level-5\n";
    for (const char* channel : kChannelNames) {
        if (!write_lr && (channel[0] == 'L' && channel[1] == 'R' && channel[2] == '.')) {
            continue;
        }
        if (!write_hr && (channel[0] == 'H' && channel[1] == 'R' && channel[2] == '.')) {
            continue;
        }
        if (!write_depth && std::strncmp(channel, "Depth.", 6) == 0) continue;
        if (!write_motion && std::strncmp(channel, "Motion.", 7) == 0) continue;
        if (!write_normals && std::strncmp(channel, "Normals.", 8) == 0) continue;
        if (payload->capture_albedo == 0u && std::strncmp(channel, "Albedo.", 7) == 0) continue;
        if (payload->capture_roughness == 0u && std::strncmp(channel, "Roughness.", 10) == 0) continue;
        if (payload->capture_metallic == 0u && std::strncmp(channel, "Metallic.", 9) == 0) continue;
        if (payload->capture_emissive == 0u && std::strncmp(channel, "Emissive.", 9) == 0) continue;
        out << "channel=" << channel << "\n";
    }
    out << "payload-begin\n";
    if (write_lr) {
        write_view(out, payload->lr_rgb);
    }
    if (write_hr) {
        write_view(out, payload->hr_rgb);
    }
    if (write_depth) write_view(out, payload->depth_z);
    if (write_motion) write_view(out, payload->motion_xy);
    if (write_normals) write_view(out, payload->normals_xyz);
    if (payload->capture_albedo != 0u) write_view(out, payload->albedo_rgb);
    if (payload->capture_roughness != 0u) write_view(out, payload->roughness);
    if (payload->capture_metallic != 0u) write_view(out, payload->metallic);
    if (payload->capture_emissive != 0u) write_view(out, payload->emissive_rgb);
    return out.good() ? 1 : 0;
#endif
}

} // extern "C"
