#include "oss_capture.h"

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
};

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

bool same_canvas(const OssCaptureFramePayload& payload) {
    const uint32_t w = payload.hr_rgb.width;
    const uint32_t h = payload.hr_rgb.height;
    return payload.lr_rgb.width == w && payload.lr_rgb.height == h &&
           payload.depth_z.width == w && payload.depth_z.height == h &&
           payload.motion_xy.width == w && payload.motion_xy.height == h &&
           payload.normals_xyz.width == w && payload.normals_xyz.height == h;
}
#endif

} // namespace

extern "C" {

int oss_capture_write_exr(const wchar_t* path, const OssCaptureFramePayload* payload) {
    if (!path || !payload) {
        return 0;
    }
    if (!valid_view(payload->lr_rgb, 3) ||
        !valid_view(payload->hr_rgb, 3) ||
        !valid_view(payload->depth_z, 1) ||
        !valid_view(payload->motion_xy, 2) ||
        !valid_view(payload->normals_xyz, 3)) {
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
    Imf::Header header(static_cast<int>(payload->hr_rgb.width), static_cast<int>(payload->hr_rgb.height));
    header.compression() = Imf::ZIP_COMPRESSION;
    for (const char* channel : kChannelNames) {
        insert_float_channel(header, channel);
    }

    Imf::FrameBuffer frame_buffer;
    insert_slice(frame_buffer, "LR.R", payload->lr_rgb, 0);
    insert_slice(frame_buffer, "LR.G", payload->lr_rgb, 1);
    insert_slice(frame_buffer, "LR.B", payload->lr_rgb, 2);
    insert_slice(frame_buffer, "HR.R", payload->hr_rgb, 0);
    insert_slice(frame_buffer, "HR.G", payload->hr_rgb, 1);
    insert_slice(frame_buffer, "HR.B", payload->hr_rgb, 2);
    insert_slice(frame_buffer, "Depth.Z", payload->depth_z, 0);
    insert_slice(frame_buffer, "Motion.X", payload->motion_xy, 0);
    insert_slice(frame_buffer, "Motion.Y", payload->motion_xy, 1);
    insert_slice(frame_buffer, "Normals.X", payload->normals_xyz, 0);
    insert_slice(frame_buffer, "Normals.Y", payload->normals_xyz, 1);
    insert_slice(frame_buffer, "Normals.Z", payload->normals_xyz, 2);

    Imf::OutputFile file(fs_path.string().c_str(), header);
    file.setFrameBuffer(frame_buffer);
    file.writePixels(static_cast<int>(payload->hr_rgb.height));
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
        out << "channel=" << channel << "\n";
    }
    out << "payload-begin\n";
    write_view(out, payload->lr_rgb);
    write_view(out, payload->hr_rgb);
    write_view(out, payload->depth_z);
    write_view(out, payload->motion_xy);
    write_view(out, payload->normals_xyz);
    return out.good() ? 1 : 0;
#endif
}

} // extern "C"
