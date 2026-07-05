#!/bin/bash

# run_tess.sh - Wrapper for TESS-4C pipeline

CONFIG_FILE="/Users/johnbarentine/Documents/Sky-brightness/TESS/Reduce-TESS-data/tess_pipeline/tess_sites.yaml"
INSTRUMENT=""
DATA_DIR=""
FLAGS=""

usage() {
    echo "Usage: ./run_tess.sh --instrument NAME [options]"
    echo ""
    echo "Required:"
    echo "  --instrument NAME     Specify instrument (e.g., stars1429)"
    echo ""
    echo "Optional:"
    echo "  --data-dir DIR        Override default path (data/NAME)"
    echo "  --sun                 Enable solar altitude filter (-18 deg)"
    echo "  --moon                Enable lunar altitude filter (-5 deg)"
    echo "  --mw                  Enable Milky Way zenith filter"
    echo "  --zod                 Enable Zodiacal light filter"
    echo "  --cloud               Enable cloud/variability filter"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instrument) 
            INSTRUMENT="$2"
            shift 2 
            ;;
        --data-dir) 
            DATA_DIR="--data-dir $2"
            shift 2 
            ;;
        --sun|--moon|--mw|--zod|--cloud) 
            FLAGS="$FLAGS $1"
            shift 
            ;;
        -h|--help) 
            usage 
            ;;
        *) 
            echo "Unknown option: $1"
            usage 
            ;;
    esac
done

if [[ -z "$INSTRUMENT" ]]; then 
    echo "Error: --instrument is required."
    usage 
fi

echo "=== TESS-4C Pipeline Runner ==="
# Note: Ensure you point this to main4c.py
CMD="python3 main4c.py --config $CONFIG_FILE --site $INSTRUMENT $DATA_DIR $FLAGS"

echo "Running: $CMD"
echo ""
eval $CMD
