cd /scratch/homes/sfan/multi_doge/src/data/datasets/climblab

CLS=5
for NUM in 003
    do
        # Define the local file path
        LOCAL_FILE="cluster_${CLS}/cluster_${CLS}_${NUM}.tokenized.parquet"
        
        # Check if the file already exists
        if [ ! -f "$LOCAL_FILE" ]; then
            echo "Downloading $LOCAL_FILE..."
            wget -O "$LOCAL_FILE" "https://huggingface.co/datasets/nvidia/ClimbLab/resolve/main/cluster_$CLS/cluster_${CLS}_${NUM}.tokenized.parquet?download=true"
        else
            echo "File $LOCAL_FILE already exists, skipping download."
        fi
    done
