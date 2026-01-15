#!/bin/bash
# scripts/clean_project.sh
# A script to reduce local project size by purging caches and downloads.

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_ROOT"

echo "🧹 Starting project cleanup..."

# 1. Purge Audio and Database Cache
if [ -f "scripts/purge_cache.py" ]; then
    echo "📦 Purging database and audio cache via Python script..."
    python3 scripts/purge_cache.py all
else
    echo "⚠️ scripts/purge_cache.py not found. Skipping DB purge."
fi

# 2. Clean Temporary Files
echo "📁 Cleaning temp/ directory..."
rm -rf temp/*
mkdir -p temp/audio_cache
touch temp/.gitkeep 2>/dev/null || true

# 3. Clean Downloaded Videos (Keyword folders)
echo "🎥 Cleaning downloaded background videos..."
# This removes all subdirectories in background_videos (which are keyword-based downloads)
if [ -d "background_videos" ]; then
    find background_videos/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
    # Also remove any direct video files in the root that aren't part of the core repo
    find background_videos/ -maxdepth 1 -name "*.mp4" -delete
    find background_videos/ -maxdepth 1 -name "*.webm" -delete
fi

# 4. Clean Output Folders
echo "📤 Cleaning output/ and backup_output/..."
rm -rf output/*
rm -rf backup_output/*
touch output/.gitkeep 2>/dev/null || true
touch backup_output/.gitkeep 2>/dev/null || true

# 5. Optional: Clean Python Build Artifacts
echo "⚙️ Cleaning Python cache/build files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "✅ DONE! Project size significantly reduced."
echo "Note: Next generation will download fresh assets as needed."
