#!/bin/bash

npm run build
# cp -r ./src/assets/* ./dist/assets # workaround
cp -r ./dist ./utu_agent_ui/static

python -m build --outdir build && echo "build success"

echo "clean up static"
rm -r ./utu_agent_ui/static