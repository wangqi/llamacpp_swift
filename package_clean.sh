#!/bin/bash

# Print commands before executing and exit on error
set -ex

echo "🧹 Cleaning Swift package..."

# Remove DerivedData
echo "Removing DerivedData..."
rm -rf ~/Library/Developer/Xcode/DerivedData/*

# Remove Package.resolved and .build directory
echo "Removing Package.resolved and .build directory..."
rm -f Package.resolved
rm -rf .build

# Clean Swift package caches
echo "Cleaning Swift package caches..."
swift package clean
swift package reset

# Rebuild package
echo "🔨 Rebuilding package..."
# Explanation of the Flags
#   • --configuration release
# Builds in release mode (optimizations enabled).
#   • --arch arm64
# Ensures the target architecture is Apple Silicon (or iOS device). You can omit or change this if you are on Intel mac, or specify a custom --destination for iOS or simulator builds. For example, --arch x86_64 if you want an Intel build.
#   • -Xcc <flag>
# Passes a flag to the C/Obj‐C compiler (Clang). Here we do:
#   • -O3 => high optimization
#   • -fno-objc-arc => (example) compile Objective‐C code without ARC
#   • -Wno-shorten-64-to-32 => suppress that particular warning
#   • -Xcxx <flag>
# Passes a flag to the C++ compiler. For example:
#   • -std=c++17 => use C++17
#   • -O3 => optimization
#   • -Xlinker <option>
# Passes options to the linker (ld). For example:
#   • -L/usr/local/lib => adds a library search path
#   • -lc++ => links the C++ standard library (on some platforms)
# 
# Do You Need All These Flags?
#   • Usually you only need swift build --configuration release and optionally -Xcc -Ofast or -Xcxx -Ofast for maximum optimization.
#   • If your code references Objective‐C frameworks (like Foundation), SwiftPM automatically adds flags to compile .m files as Objective‐C.
#   • The sample above shows how to pass additional fine-grained flags if you want them.
# 
swift build --configuration release \
  --arch arm64 \
  -Xcc -O3 \
  -Xcc -fno-objc-arc \
  -Xcc -Wno-shorten-64-to-32 \
  -Xcxx -std=c++17 \
  -Xcxx -O3 \
  -Xlinker -L/usr/local/lib \
  -Xlinker -lc++

echo "✅ Done! Package has been cleaned and rebuilt."
