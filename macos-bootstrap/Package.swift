// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AutocompleterBootstrap",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "AutocompleterBootstrap",
            path: "Sources/AutocompleterBootstrap"
        )
    ]
)
