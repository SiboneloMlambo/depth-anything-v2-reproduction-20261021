#!/usr/bin/env bash
# Clone the two official repositories at pinned commits into external/.
# Model definitions and inference code come from here; nothing else does.
set -euo pipefail
cd "$(dirname "$0")/.."

V1_URL=https://github.com/LiheYoung/Depth-Anything.git
V1_COMMIT=1d03336771fe09c5398ffdd211441e33941a97dc
V2_URL=https://github.com/DepthAnything/Depth-Anything-V2.git
V2_COMMIT=a561b849ebae10a6f5ef49e26c83cbbcd36c71bf

clone_pinned () {
  local url=$1 commit=$2 dir=$3
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
  git -C "$dir" checkout --quiet "$commit"
  echo "[setup_external] $dir at $(git -C "$dir" rev-parse HEAD)"
}

mkdir -p external
clone_pinned "$V1_URL" "$V1_COMMIT" external/Depth-Anything
clone_pinned "$V2_URL" "$V2_COMMIT" external/Depth-Anything-V2
