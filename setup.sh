#!/bin/bash
# setup.sh

# Create directories
mkdir -p models/
mkdir -p data/lobster/

# Download model file (replace with actual URL)
wget https://your-model-repo/abuse_detector.h5 -O models/abuse_detector.h5

echo "Setup complete!"
