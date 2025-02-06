#!/bin/bash

# Define the directory where Python files will be created
CREATION_PATH="/home/calatrava/Documents/PhD/Thesis/other_works/UniMTS/datasets"

# Define the list of datasets
DATASETS=(
    "PAMAP2"
    "MHEALTH"
    "REALDISP"
    "UCIHAR"
    "HARTH"
    "DSADS"
    "WISDM"
    "USCHAD"
    "OPPORTUNITY"
    "WHARF"
    "UTDMHAD"
    "MOTIONSENSE"
    "WHAR"
    "SHOAIB"
    "HAR70PLUS"
    "REALWORLD"
    "TNDAHAR"
    "UTCOMPLEX"
)

# Create the directory if it doesn't exist
mkdir -p "$CREATION_PATH"

# Loop through each dataset and create a Python file
for DATASET in "${DATASETS[@]}"; do
    PYTHON_FILE="$CREATION_PATH/${DATASET}.py"

    # Create the Python file with a simple structure
    cat <<EOF > "$PYTHON_FILE"
# ${DATASET}.py - Script for handling ${DATASET} dataset

def download_${DATASET,,}():
    \"\"\" Function to download and process the ${DATASET} dataset \"\"\"
    print("Downloading ${DATASET} dataset...")

if __name__ == "__main__":
    download_${DATASET,,}()
EOF

    echo "✅ Created: $PYTHON_FILE"
done

echo "🎯 All Python scripts created in $CREATION_PATH"
