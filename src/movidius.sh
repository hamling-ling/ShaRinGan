
rm -rf last_node_name.tx
python sharingan_save_for_movidius.py \
  --output_dir "./" \
  --checkpoint sharingan_checkpoints

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

rm -rf graph
mvNCCompile movidius.meta.meta -in=input -on $NODE_NAME -s 12
if [ $? -gt 0 ]; then
    echo "error: mvNCCompile failed"
    exit 1
fi
if [ ! -e graph ]; then
    echo "error:graph not found"
    exit 1
fi
echo "graph created!"

python run_movidius_singleshot.py
if [ $? -gt 0 ]; then
    echo "error: run_movidius_singleshot.py failed"
    exit 1
fi
