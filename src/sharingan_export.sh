#!/bin/bash

# Where the chepoint file is.
CHEKPOINT_FILE="../data/output/quantization_aware_model/model.ckpt-95850"

echo -------- exporting --------
rm -rf ../data/output/frozen_model
python sharingan_export_test.py \
  --output_dir "../data/output/frozen_model" \
  --checkpoint $CHEKPOINT_FILE

echo -------- freezing --------
python -m tensorflow.python.tools.freeze_graph \
--input_graph="../data/output/frozen_model/inference_graph.pb" \
--input_checkpoint=${CHEKPOINT_FILE} \
--input_binary=true \
--output_graph="../data/output/frozen_model/frozen.pb" \
--output_node_names=generator/Tanh

echo -------- converting to tflite --------
convert_to_fp_tflite()
{
  FORMAT=$1
  OUT_FILE=$2
  echo converting $FORMAT from $OUT_FILE
  tflite_convert --graph_def_file="../data/output/frozen_model/frozen.pb" \
  --output_file=$OUT_FILE \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=$FORMAT \
  --inference_type=FLOAT \
  --output_arrays=generator/Tanh \
  --input_arrays=input \
  --mean_values 121 \
  --std_dev_values=64
}

convert_to_ui8_tflite()
{
  FORMAT=$1
  OUT_FILE=$2
  echo converting $FORMAT from $OUT_FILE
  tflite_convert --graph_def_file="../data/output/frozen_model/frozen.pb" \
  --output_file=$OUT_FILE \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=$FORMAT \
  --inference_type=QUANTIZED_UINT8 \
  --output_arrays=generator/Tanh \
  --input_arrays=input \
  --mean_values 121 \
  --std_dev_values=64 \
  --default_ranges_min=0 \
  --default_ranges_max=6
}

#convert_to_ui8_tflite TFLITE ../data/output/frozen_model/sharingan_ui8.tflite
#convert_to_ui8_tflite GRAPHVIZ_DOT "../data/output/frozen_model/sharingan_ui8.dot"

convert_to_fp_tflite TFLITE ../data/output/frozen_model/sharingan_fp.tflite
convert_to_fp_tflite GRAPHVIZ_DOT "../data/output/frozen_model/sharingan_fp.dot"

#how to convert
#dot -Tpdf -O /tmp/foo.dot
