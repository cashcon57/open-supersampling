// OSS-Gaussian Metal rasterizer Swift host harness — Sprint 7 / T7.M.1 skeleton.
//
// Loads `rasterizer.metallib`, looks up the `gaussian_rasterize_tile` compute
// function, and exposes a single `dispatch(...)` entry point. Used by the
// Sprint 7 unit harness for kernel parity testing against the Python
// reference rasterizer.
//
// The Python production driver (`run_sintel.py`) calls into Swift via PyObjC;
// this file is the underlying Swift implementation.
//
// Build:
//     swiftc -O rasterizer.swift -o rasterizer_test \
//         -framework Metal -framework Foundation
//
// Sprint 7 prep ships the dispatch skeleton only. T7.M.2 fills out the
// staging-buffer marshalling once the kernel body lands.

import Foundation
import Metal

public struct DispatchParams {
    public var numGaussians: UInt32
    public var outH: UInt32
    public var outW: UInt32
    public var featDim: UInt32
    public var topK: UInt32
    public var pad0: UInt32 = 0
    public var pad1: UInt32 = 0
    public var pad2: UInt32 = 0
}

public enum GaussianRasterizerError: Error {
    case noDevice
    case missingMetallib(String)
    case missingFunction(String)
    case pipelineFailed(String)
}

public final class GaussianRasterizer {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    public init(metallibURL: URL) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GaussianRasterizerError.noDevice
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw GaussianRasterizerError.pipelineFailed("makeCommandQueue")
        }
        self.queue = queue

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(URL: metallibURL)
        } catch {
            throw GaussianRasterizerError.missingMetallib(metallibURL.path)
        }

        guard let fn = library.makeFunction(name: "gaussian_rasterize_tile") else {
            throw GaussianRasterizerError.missingFunction("gaussian_rasterize_tile")
        }

        do {
            self.pipeline = try device.makeComputePipelineState(function: fn)
        } catch {
            throw GaussianRasterizerError.pipelineFailed(String(describing: error))
        }
    }

    /// Dispatch a single tile rasterization pass. Buffers must be pre-populated
    /// by the caller with the layout documented in `rasterizer.metal`.
    /// TODO(T7.M.2): wire the buffer marshalling once the kernel lands.
    public func dispatch(gaussians: MTLBuffer,
                        tileIndex: MTLBuffer,
                        tileStarts: MTLBuffer,
                        outImage: MTLBuffer,
                        params: DispatchParams) throws {
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw GaussianRasterizerError.pipelineFailed("encoder")
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(gaussians, offset: 0, index: 0)
        enc.setBuffer(tileIndex, offset: 0, index: 1)
        enc.setBuffer(tileStarts, offset: 0, index: 2)
        enc.setBuffer(outImage, offset: 0, index: 3)

        var p = params
        enc.setBytes(&p, length: MemoryLayout<DispatchParams>.size, index: 4)

        let tileSize = 16
        let groups = MTLSize(width: (Int(params.outW) + tileSize - 1) / tileSize,
                             height: (Int(params.outH) + tileSize - 1) / tileSize,
                             depth: 1)
        let threads = MTLSize(width: tileSize, height: tileSize, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
}
