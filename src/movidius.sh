#!/bin/bash

echo ""
echo "################################"
echo " Saving Checkpoints"
echo "################################"
echo ""
rm -rf last_node_name.txt
python3 sharingan_save_for_movidius.py \
  --output_dir "../data/output/movidius" \
  --checkpoint "../data/output/pretrained_model/model.ckpt-8000"

if [ $? -gt 0 ]; then
    echo "error: sharingan_save_for_movidius.py failed"
    exit 1
fi
if [ ! -e last_node_name.txt ]; then
    echo "error: last_node_name.txt not found"
    exit 1
fi

NODE_NAME=$(cat last_node_name.txt)
echo "NODE_NAME=$NODE_NAME"

echo ""
echo "################################"
echo " Compiling Graph"
echo "################################"
echo ""
GRAPH_PATH=../data/output/movidius/graph
rm -rf GRAPH_PATH
mvNCCompile ../data/output/movidius/movidius.meta -in=input -on $NODE_NAME -s 12 -o $GRAPH_PATH
if [ $? -gt 0 ]; then
    echo "error: mvNCCompile failed"
    exit 1
fi
if [ ! -e $GRAPH_PATH ]; then
    echo "error:graph not found"
    exit 1
fi
echo "graph created! $GRAPH_PATH"

echo ""
echo "################################"
echo " running inference"
echo "################################"
echo ""
python3 run_movidius_singleshot.py \
    --graph "../data/output/movidius/graph" \
    --input "../data/input/evaluation/002000.bin" \
    --output_dir "../data/output/movidius/image"
if [ $? -gt 0 ]; then
    echo "error: run_movidius_singleshot.py failed"
    exit 1
fi
