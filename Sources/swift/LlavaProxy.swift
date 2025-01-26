//
//  LlavaProxy.swift
//
//  A Swift "proxy" that wraps the functions from lava.h and clip.h
//  so that LLaMaMM_MModal.swift no longer needs to import those headers directly.
//

import Foundation
import llama

/// A Swift wrapper for `struct clip_ctx *`.
/// We keep it as an opaque pointer in Swift.
public final class ClipContext {
    var pointer: OpaquePointer?
    public init(_ pointer: OpaquePointer?) {
        self.pointer = pointer
    }
}

/// A Swift wrapper for `struct llava_image_embed`.
/// We store a pointer as well.
public struct LlavaImageEmbed {
    public var pointer: UnsafeMutablePointer<llava_image_embed>?
    public init(pointer: UnsafeMutablePointer<llava_image_embed>?) {
        self.pointer = pointer
    }
}

// MARK: - Clip Model

/// Loads a CLIP model from file using `clip_model_load(...)`.
/// - Parameters:
///   - filename: The path to the model on disk.
///   - verbosity: Typically 0 or 1 to control logging.
/// - Returns: A `ClipContext` wrapper, or nil on failure.
public func llava_clip_model_load(filename: String, verbosity: Int32) -> ClipContext? {
    guard let cString = filename.cString(using: .utf8) else { return nil }
    // Bridging header must declare: `CLIP_API struct clip_ctx * clip_model_load(const char * fname, int verbosity)`
    guard let ptr = clip_model_load(cString, verbosity) else {
        return nil
    }
    return ClipContext(ptr)
}

/// Frees a CLIP model using `clip_free(...)`.
public func llava_clip_free(_ ctx: ClipContext?) {
    guard let c = ctx, let ptr = c.pointer else { return }
    clip_free(ptr)
    c.pointer = nil
}

// MARK: - Image Embeddings

/// Builds an image embed from an image file path via `llava_image_embed_make_with_filename(...)`.
/// - Parameters:
///   - clipCtx: The `ClipContext` of the loaded CLIP model
///   - nThreads: Number of threads
///   - imagePath: Path to image
/// - Returns: A `LlavaImageEmbed` containing pointer, or nil if something fails
public func llava_image_embed_make_with_filename(
    clipCtx: ClipContext?,
    nThreads: Int32,
    imagePath: String
) -> LlavaImageEmbed? {
    guard let cClip = clipCtx?.pointer else { return nil }
    guard let cStr = imagePath.cString(using: .utf8) else { return nil }

    // bridging header must declare:
    //   LLAVA_API struct llava_image_embed * llava_image_embed_make_with_filename(
    //       struct clip_ctx * ctx_clip,
    //       int n_threads,
    //       const char * image_path
    //   );
    let ptr = llava_image_embed_make_with_filename(cClip, nThreads, cStr)
    guard ptr != nil else { return nil }
    return LlavaImageEmbed(pointer: ptr)
}

/// Frees the memory for an image embedding via `llava_image_embed_free(...)`.
public func llava_image_embed_free(_ embed: LlavaImageEmbed?) {
    guard let e = embed?.pointer else { return }
    llava_image_embed_free(e)
}

/// Evaluate the image embedding by writing the embed tokens into
/// the llama context with batch size `nBatch`, starting at `nPast`.
/// - bridging header must declare:
///     LLAVA_API bool llava_eval_image_embed(struct llama_context * ctx_llama,
///                                          const struct llava_image_embed * embed,
///                                          int n_batch,
///                                          int * n_past);
///
/// - Parameters:
///   - ctxLlama: pointer to a llama_context
///   - embed: pointer to llava_image_embed
///   - nBatch: batch size
///   - nPast: pointer to an integer that will be updated
/// - Returns: Bool result from C call
public func llava_eval_image_embed(
    ctxLlama: OpaquePointer?,
    embed: LlavaImageEmbed?,
    nBatch: Int32,
    nPast: inout Int32
) -> Bool {
    guard let cEmbed = embed?.pointer, let cCtx = ctxLlama else { return false }
    var localPast = nPast
    let ok = llava_eval_image_embed(cCtx, cEmbed, nBatch, &localPast)
    nPast = localPast
    return ok
}

// MARK: - (Optional) Additional llava or clip calls
// If you need more calls from lava.h or clip.h, wrap them here similarly.
