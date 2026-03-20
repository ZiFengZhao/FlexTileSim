import argparse
import csv
import math

from gemm import gen_inst

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ScaleSim CSV to GEMM m/n/k for multi-tile simulator")
    parser.add_argument("--model", type=str, required=True, help="Target DNN model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to ScaleSim CSV topology file")
    parser.add_argument("--tile_num", type=int, default=1, help="Number of tiles (for layer splitting)")
    return parser.parse_args()

def transform(layer):
    layer_name = layer['Layer name']
    H_in = int(layer['IFMAP Height'])
    W_in = int(layer['IFMAP Width'])
    R = int(layer['Filter Height'])
    S = int(layer['Filter Width'])
    C_in = int(layer['Channels'])
    C_out = int(layer['Num Filter'])
    stride = int(layer['Strides'])

    H_out = math.ceil(H_in / stride)
    W_out = math.ceil(W_in / stride)

    k = C_in * R * S
    n = H_out * W_out
    m = C_out

    return layer_name, m, n, k, H_out, W_out

def split_layer(layer_name, m, n, k, tile_num):
    base_m = m // tile_num
    remainder = m % tile_num
    result = []
    for i in range(tile_num):
        tile_m = base_m + (1 if i < remainder else 0)
        tile_name = layer_name
        result.append((tile_name, tile_m, n, k))
    return result

def main():
    args = parse_args()

    num_cores = args.tile_num
    num_ddr_channels = -1
    if num_cores == 1 or num_cores == 2:
        num_ddr_channels = 1
    elif num_cores == 4 or num_cores == 8:
        num_ddr_channels = 2
    elif num_cores == 16:
        num_ddr_channels = 4
    elif num_cores == 32 or num_cores == 64:
        num_ddr_channels = 8
    else:
        raise ValueError("Invalid number of cores")

    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        layers = [{k.strip(): v.strip() for k,v in row.items()} for row in reader]

    all_core_instrs = { core_id: [] for core_id in range(args.tile_num) }

    prev_layer_instr_id = []
    for i in range(args.tile_num):
        prev_layer_instr_id.append(0)

    for layer in layers:
        csv_line = ','.join([layer[h.strip()] for h in layer.keys()])

        layer_name, m, n, k, H_out, W_out = transform(layer)

        mnk_per_tile = split_layer(layer_name, m, n, k, args.tile_num)
        tile_mnk_str = ', '.join([f"Tile {i}: m={tile_m}, n={n}, k={k}" for i, (_, tile_m, n, k) in enumerate(mnk_per_tile)])

        for tile_id, (_, tile_m, n, k) in enumerate(mnk_per_tile):
            
            instrs = gen_inst(
                M = tile_m,
                N = n,
                K = k,
                sa_size = 32,
                core_id = tile_id,
                num_cores= args.tile_num,
                num_ddr_channels = num_ddr_channels,
                instr_id = prev_layer_instr_id[tile_id]
            )

            all_core_instrs[tile_id].extend(instrs)
            current_inst_id = prev_layer_instr_id[tile_id] + len(instrs) -1
            if args.tile_num > 1:
                sync_inst = f"Sync {len(instrs)} dep=00 sync_id={current_inst_id+1} sync_core_count={args.tile_num} // [Gemm] Sync"
                all_core_instrs[tile_id].append(sync_inst)
            prev_layer_instr_id[tile_id] = current_inst_id


    out_inst_file = args.model + "_"+ (str)(args.tile_num) + "tile" + "_inst.txt"
    with open(out_inst_file, "w") as f:

        for core_id in range(args.tile_num):

            f.write(f"# Core {core_id} instructions\n")

            for inst in all_core_instrs[core_id]:
                f.write(str(inst) + "\n")

            f.write("\n")
    print(f"{out_inst_file} generated successfully.")


if __name__ == "__main__":
    main()