#!/bin/bash

if [ "$1" = "cleanup" ]; then
    rm *.png
    rm *.jpg
    rm *.jpeg
    rm *.gif
    rm *.webp
    rm *.svg
    rm *.pptx
    rm *.json
    exit 0
fi

# if there is $2
if [ -n "$2" ]; then
    if [ "$2" = "fill" ]; then
        python fill_template.py \
            -t templates \
            -n $1 \
            -i tt3.json  \
        -o tt3.pptx \
        --yaml_config yaml_example2.yaml && \
        open tt3.pptx
        exit 0
    fi
fi

python main.py \
  --file webpage.html \
  --template_path templates \
  --yaml_path yaml_example2.yaml \
  --pages 8 \
  --disable_tooluse \
  --extra_prompt "Ensure rich content. In English" \
  --template_name $1 \
  --output_json tt3.json \
  --output_path tt3.pptx && \
open tt3.pptx