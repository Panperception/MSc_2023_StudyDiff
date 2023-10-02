#!/bin/bash

# 路径到目标文件夹
DIR="/root/autodl-tmp/trained_models/oxford_flower"

# 遍历文件夹中的每一个文件
for file in "$DIR"/*; do
    # 使用basename获取没有路径的文件名
    base=$(basename "$file")
    # 使用awk分割文件名并获取下划线后的部分
    new_name=$(echo "$base" | awk -F'_' '{print $NF}')
    # 如果新文件名与旧文件名不同，则重命名
    if [ "$new_name" != "$base" ]; then
        mv "$file" "$DIR/$new_name"
    fi
done



find /root/autodl-tmp/generated_img/ -type f -name "*.png" -exec rm -f {} \;
