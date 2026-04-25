#!/usr/bin/env python3
"""Download results from Modal volume."""

import modal
import json
from pathlib import Path

# Initialize Modal
app = modal.App("download-results")
volume = modal.Volume.from_name("feedback-geometry-data", create_if_missing=False)

@app.function(volumes={"/data": volume})
def download_results():
    """Download experiment results from Modal volume."""
    import os
    
    results = {}
    
    # Check what's in the volume
    print("Volume contents:")
    for root, dirs, files in os.walk("/data"):
        level = root.replace("/data", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
            
            # Read JSON files
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        results[filepath] = data
                        print(f"{subindent}  -> Loaded {len(str(data))} bytes")
                except Exception as e:
                    print(f"{subindent}  -> Error: {e}")
    
    return results

@app.local_entrypoint()
def main():
    results = download_results.remote()
    
    # Save locally
    output_dir = Path("results/modal_downloads")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for path, data in results.items():
        # Create relative path
        rel_path = path.replace("/data/", "")
        local_path = output_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved: {local_path}")
    
    print(f"\nAll results saved to {output_dir}")
    
    # Print summaries
    if results:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        for path, data in results.items():
            print(f"\n{path}:")
            if "summary" in data:
                for key, val in data["summary"].items():
                    print(f"  {key}: {val}")
