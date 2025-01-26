// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

var cSettings: [CSetting] =  [
    .define("SWIFT_PACKAGE"),
    .define("GGML_USE_ACCELERATE"),
    .define("GGML_BLAS_USE_ACCELERATE"),
    .define("ACCELERATE_NEW_LAPACK"),
    .define("ACCELERATE_LAPACK_ILP64"),
    .define("GGML_USE_BLAS"),
    .define("GGML_USE_LLAMAFILE"),
    .define("GGML_METAL_NDEBUG"),
    .define("NDEBUG"),
    .define("GGML_USE_CPU"),
    .define("GGML_USE_METAL"),
    
    .unsafeFlags(["-Ofast"], .when(configuration: .release)),
    .unsafeFlags(["-O3"], .when(configuration: .debug)),
    .unsafeFlags(["-mfma","-mfma","-mavx","-mavx2","-mf16c","-msse3","-mssse3"]), //for Intel CPU
    .unsafeFlags(["-pthread"]),
    .unsafeFlags(["-fno-objc-arc"]),
    .unsafeFlags(["-Wno-shorten-64-to-32"]),
    .unsafeFlags(["-fno-finite-math-only"], .when(configuration: .release)),
    .unsafeFlags(["-w"]),    // ignore all warnings
    
    //header search path
    .headerSearchPath("include"),
]

var linkerSettings: [LinkerSetting] = [
    .linkedFramework("Foundation"),
    .linkedFramework("Accelerate"),
    .linkedFramework("Metal"),
    .linkedFramework("MetalKit"),
    .linkedFramework("MetalPerformanceShaders"),
]

let package = Package(
    name: "llmfarm_core",
    platforms: [.macOS(.v12),.iOS(.v15)],
    products: [
        .library(
            name: "llmfarm_core",
            targets: ["llmfarm_core"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/wangqi/llama.cpp.git", branch: "master")
    ],
    targets: [
        .target(
            name: "llmfarm_core_cpp",
            dependencies: [
              .product(name: "llama", package: "llama.cpp")
            ],
            path: "Sources/llmfarm_core_cpp",
            exclude: ["spm-headers"],
            sources: [
                "exception_helper_objc.mm", 
                //"exception_helper.cpp",
            ],
            publicHeadersPath: "include",
            cSettings: cSettings,
            linkerSettings: linkerSettings
        ),
        .target(
              name: "llmfarm_core",
              dependencies: [
                "llmfarm_core_cpp",
                .product(name: "llama", package: "llama.cpp")
              ],
              path: "Sources/llmfarm_core"
        )
    ],
    
    cxxLanguageStandard: .cxx17
)
