## Actually here is pretraning script, not training script


base_data_dir=/home/jincan/long_seg/my3DUNet/Data/Imaging_for_3DUNet_long
LR=1e-4                   # 学习率
MAX_EPOCH=800             # 最大训练轮数
EVAL_FREQ=10              # 评估频率
OUTPUT_DIR="./output"     # 输出目录
GPU_ID="1"                # GPU ID
CLASSES=4                 # 分类数量
MOD="long_seg"            # 模型类型，可选择"long_seg"或"single_seg"

# 运行 Python 脚本，并传递参数
python ../train.py \
    --lr $LR \
    --max_epoch $MAX_EPOCH \
    --eval_freq $EVAL_FREQ \
    --output_dir $OUTPUT_DIR \
    --gpu_id $GPU_ID \
    --classes $CLASSES \
    --mod $MOD \
    --train_path $base_data_dir/train \
    --val_path $base_data_dir/val \
    
