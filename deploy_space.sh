#!/usr/bin/env bash
# Publish the current main branch as the public website on Hugging Face.
#
#   bash deploy_space.sh
#
# The Space uses the free Gradio SDK hardware, but runs our own Flask site:
# app_file points at web/app.py, which serves on port 7860 when it sees it is
# on a Space. packages.txt supplies the system libraries OpenCV and MediaPipe
# need. (The Dockerfile is kept for any host that runs containers.)
#
# Hugging Face reads the Space's settings from a header at the top of
# README.md, which the GitHub README should not show. So this builds a
# throwaway commit on top of main with the header added and force-pushes it
# to the Space's main branch. Your own branches are left exactly as they were.
#
# The first time, git asks for your Hugging Face username and an access token
# with write permission (huggingface.co -> Settings -> Access Tokens). Windows
# remembers it after that.
set -euo pipefail
cd "$(dirname "$0")"
REMOTE=${1:-space}

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Commit or stash your changes first."; exit 1
fi

here=$(git rev-parse --abbrev-ref HEAD)
cleanup() { git checkout -q "$here"; git branch -D hf-deploy >/dev/null 2>&1 || true; }
trap cleanup EXIT

git checkout -q -B hf-deploy main
{
cat <<'EOF'
---
title: Delivery Stride
emoji: 🏏
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.32.0"
python_version: "3.13"
app_file: web/app.py
pinned: false
short_description: Fast bowling technique from one side-on phone video
---

EOF
cat README.md
} > README.space.md
mv README.space.md README.md
git commit -q -am "Hugging Face Space configuration"
git push --force "$REMOTE" hf-deploy:main

echo
echo "Pushed. The Space rebuilds in 5-10 minutes:"
echo "  $(git remote get-url "$REMOTE" | sed 's#\.git$##')"
