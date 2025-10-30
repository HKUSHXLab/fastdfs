#!/usr/bin/env bash
set -euo pipefail

# Change to the directory of this script to ensure relative paths work
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

echo "Running minimal_examples create_example.py scripts..."

# Find example directories and execute their create_example.py in numeric order
# Expected directories: example1_*, example2_*, ...
examples=(
  example1_direct_features
  example2_single_aggregation
  example3_multi_level_aggregation
  example4_cutoff_time
  example5_multiple_relationships
)

for ex in "${examples[@]}"; do
  if [[ -f "$SCRIPT_DIR/$ex/create_example.py" ]]; then
    echo "\n==> Running: $ex/create_example.py"
    python "$SCRIPT_DIR/$ex/create_example.py"
  else
    echo "\n[Skip] $ex/create_example.py not found"
  fi
done

echo "\n==> Running: run_all_examples.py"
python "$SCRIPT_DIR/run_all_examples.py"

echo "\nAll minimal_examples completed successfully."


