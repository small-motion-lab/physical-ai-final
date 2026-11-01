#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAI_NAME="${PAI_CONTAINER_NAME:-pai_ros2}"
PAI_PORT="${PAI_DESKTOP_PORT:-6080}"
IMAGE='airobotbook/ros2-desktop-ai-robot-book-humble@sha256:b58ec74c019aacd9ce4818fa25fc5a9e0a1f22f8c9cd1f5260994bf15cd9eb63'
if docker inspect "$PAI_NAME" >/dev/null 2>&1; then
  actual="$(docker inspect "$PAI_NAME" --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}')"
  if [[ "$actual" != "$ROOT" ]]; then
    echo 'pai_ros2 belongs to another workspace; refusing to reuse it.' >&2
    exit 1
  fi
  docker start "$PAI_NAME" >/dev/null
else
  docker run -d --name "$PAI_NAME" --platform linux/amd64 --shm-size=512m \
    -p "127.0.0.1:${PAI_PORT}:80" \
    -v "$ROOT:/workspace" "$IMAGE"
fi
printf 'Robot desktop: http://127.0.0.1:%s/\n' "$PAI_PORT"
