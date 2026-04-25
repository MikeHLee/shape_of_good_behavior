#!/bin/bash
# Download results from Modal volume
# Usage: ./download_modal_results.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="../results"
mkdir -p "$OUTPUT_DIR"

echo "Downloading results from Modal volume 'feedback-geometry-data'..."

# Download H1 exploitation results
echo "  - H1 exploitation results..."
modal volume get feedback-geometry-data h1_exploitation/ "$OUTPUT_DIR/h1_exploitation/" 2>/dev/null || echo "    (no results yet)"

# Download sandbagging results
echo "  - Sandbagging v2 results..."
modal volume get feedback-geometry-data sandbagging_v2/ "$OUTPUT_DIR/sandbagging_v2/" 2>/dev/null || echo "    (no results yet)"

echo ""
echo "Done! Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"
