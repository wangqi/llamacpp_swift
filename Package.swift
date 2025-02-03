// swift-tools-version: 5.8
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
    name: "llamacpp_swift",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(
            name: "llamacpp_swift",
            targets: ["llamacpp_swift"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/wangqi/llama.cpp.git", branch: "master"),
        .package(url: "https://github.com/wangqi/Jinja", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.1"),
    ],
    targets: [
        .target(
            name: "llamacpp_swift_cpp",
            dependencies: [
              .product(name: "llama", package: "llama.cpp")
            ],
            path: "Sources/cpp",
            sources: [
                "exception_helper_objc.mm", 
                "package_helper.m", 
                //"exception_helper.cpp",
            ],
            publicHeadersPath: "include",
            cSettings: cSettings,
            cxxSettings: [
                .unsafeFlags(["-frtti"])  // Enable RTTI
            ],
            linkerSettings: linkerSettings
        ),
        .target(
              name: "llamacpp_swift",
              dependencies: [
                "llamacpp_swift_cpp",
                .product(name: "llama", package: "llama.cpp"),
                .product(name: "Jinja", package: "Jinja"),
                .product(name: "Logging", package: "swift-log"),
              ],
              path: "Sources/swift"
        )
    ],
    
    cxxLanguageStandard: .cxx17
)
