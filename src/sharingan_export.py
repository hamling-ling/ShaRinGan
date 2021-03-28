from sharingan_base import *
from collections import namedtuple
from tensorflow.python.platform import gfile

SZ = 1024

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or ckpt path (ex sharingan_checkpoints/model.ckpt-1001) to use for testing")

    a = parser.parse_args()

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if(os.path.isdir(a.checkpoint)):
        dir_cp = a.checkpoint
    else:
        dir_cp = os.path.dirname(a.checkpoint)

    filename = os.path.join(dir_cp, "hyper_params.json")
    with open(filename) as fd:
        json_str = fd.read()
        print("hyper parameters=\n", json_str)
        hyp = json.loads(json_str, object_hook=lambda d: namedtuple('HyperParams', d.keys())(*d.values()))

    return a, hyp

def save_last_node_name(node_name):
    f = open('last_node_name.txt', 'w')
    f.write(node_name)
    f.close()
    print(node_name, " saved in last_node_name.txt")

def main():
    a, hyper_params = process_args()

    with tf.Graph().as_default() as graph:
        input = tf.placeholder("float", [1, 1, SZ, 1], name="input")
        with tf.variable_scope("generator"):
            generator = create_generator(generator_inputs           = input,
                                         generator_outputs_channels = 1,
                                         hyper_params               = hyper_params,
                                         is_training                = False,
                                         is_fused                   = False)

        # save graph that not including weights
        filename = "inference_graph.pb"
        if(hyper_params.enable_quantization):
            print("exporting quantized graph")
            tf.contrib.quantize.create_eval_graph()
        else:
            print("exporting on-quantized graph")

        graph_def = graph.as_graph_def()
        outfile = os.path.join(a.output_dir, filename)
        with gfile.GFile(outfile, 'wb') as f:
            f.write(graph_def.SerializeToString())
        print(outfile, "saved")
main()
