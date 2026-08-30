// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ANE-LM",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "ANELMRuntime", targets: ["ANELMRuntime"])
    ],
    targets: [
        .target(
            name: "ANELMRuntime",
            path: ".",
            sources: [
                "common.cpp",
                "core/ane_runtime.cpp",
                "core/cpu_ops.cpp",
                "core/model_loader.cpp",
                "core/safetensors.cpp",
                "core/sampling.cpp",
                "models/llm/qwen3.cpp",
                "swift/ane_lm_runtime.cpp"
            ],
            publicHeadersPath: "swift/include",
            cxxSettings: [
                .define("ACCELERATE_NEW_LAPACK"),
                .headerSearchPath("."),
                .headerSearchPath("include"),
                .headerSearchPath("vendor/nlohmann/single_include")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("IOSurface"),
                .linkedLibrary("objc")
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)
