#!/bin/bash
# Warehouse Picklist Optimization Pipeline Runner
# Usage: ./run_pipeline.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=================================================="
echo "🏭 WAREHOUSE PICKLIST OPTIMIZATION PIPELINE"
echo "=================================================="
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate conda environment
echo -e "${YELLOW}Activating conda environment: warehouse_opt${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate warehouse_opt

# Check Python version
echo -e "${GREEN}Python version:${NC}"
python --version

# Run the pipeline
echo -e "\n${BLUE}Starting optimization pipeline...${NC}\n"

python main.py "$@"

echo -e "\n${GREEN}Pipeline execution complete!${NC}"
echo -e "${BLUE}==================================================${NC}"
