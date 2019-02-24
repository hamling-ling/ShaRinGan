#!/bin/bash

echo ""
echo "################################"
echo " Saving Checkpoints"
echo "################################"
echo ""

# run following in tensorflow environment
#rm -rf last_node_name.txt
#./sharingan_export.sh
#if [ $? -gt 0 ]; then
#    echo "error: sharingan_export.sh failed"
#    exit 1
#fi
#if [ ! -e last_node_name.txt ]; then
#    echo "error: last_node_name.txt not found"
#    exit 1
#fi

NODE_NAME=$(cat last_node_name.txt)
echo "NODE_NAME=$NODE_NAME"

echo ""
echo "################################"
echo " Compiling Graph"
echo "################################"
echo ""
GRAPH_PATH=../data/output/frozen_model/graph
rm -rf GRAPH_PATH
mvNCCompile ../data/output/frozen_model/movidius.meta -in=input -on $NODE_NAME -s 12 -o $GRAPH_PATH
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
    --graph "../data/output/frozen_model/graph" \
    --input "../data/input/evaluation/002000.bin" \
    --output_dir "../data/cvt/frozen_model/image"
if [ $? -gt 0 ]; then
    echo "error: run_movidius_singleshot.py failed"
    exit 1
fi
