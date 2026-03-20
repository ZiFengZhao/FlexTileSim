import argparse
import csv
import math
import struct
import re
import codegen
from gemm import gen_inst

INSTRUCTION_REGEX = codegen.INSTRUCTION_REGEX
FIELD_REGEX = codegen.FIELD_REGEX

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

    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        layers = [{k.strip(): v.strip() for k,v in row.items()} for row in reader]

    all_core_instrs = { core_id: [] for core_id in range(args.tile_num) }

    prev_layer_instr_id = []
    for i in range(args.tile_num):
        prev_layer_instr_id.append(0)

    for layer in layers:
        csv_line = ','.join([layer[h.strip()] for h in layer.keys()])
        print(f"Original: {csv_line}")

        layer_name, m, n, k, H_out, W_out = transform(layer)
        print(f"GEMM (m/n/k): {layer_name}, m={m}, n={n}, k={k}")

        mnk_per_tile = split_layer(layer_name, m, n, k, args.tile_num)
        tile_mnk_str = ', '.join([f"Tile {i}: m={tile_m}, n={n}, k={k}" for i, (_, tile_m, n, k) in enumerate(mnk_per_tile)])
        print(f"Split among {args.tile_num} tiles: {tile_mnk_str}\n")

        for tile_id, (_, tile_m, n, k) in enumerate(mnk_per_tile):
            
            instrs = gen_inst(
                M = tile_m,
                N = n,
                K = k,
                sa_size = 32,
                core_id = tile_id,
                num_cores= args.tile_num,
                instr_id = prev_layer_instr_id[tile_id]
            )

            for line in instrs:
                instr = parse_instruction_string(line)
                if instr is None:
                    continue
                instr_name = instr['instruction'].upper()
                machine_code = codegen.ENCODERS[instr_name](instr)
                all_core_instrs[tile_id].append(machine_code)
            
            current_inst_id = prev_layer_instr_id[tile_id] + len(instrs) -1
            sync_instr = {
                'instruction': 'SYNC',
                'id': len(instrs),
                'sync_core_count': args.tile_num,
                'sync_id': current_inst_id + 1
            }

            machine_code = codegen.ENCODERS['SYNC'](sync_instr)

            all_core_instrs[tile_id].append(machine_code)
            prev_layer_instr_id[tile_id] = current_inst_id

    out_file = args.model + "_"+ (str)(args.tile_num) + "tile" + "_inst.bin"
    index_entry_format = '<QI'
    index_entry_size = struct.calcsize(index_entry_format)
    instruction_size = 16

    max_core_id = args.tile_num - 1

    data_area_start_offset = (max_core_id + 1) * index_entry_size
    with open(out_file, 'wb') as f:

        index_entries = []
        current_data_offset = data_area_start_offset

        for core_id in range(args.tile_num):

            codes = all_core_instrs[core_id]

            instruction_count = len(codes)

            entry = struct.pack(index_entry_format,
                                current_data_offset,
                                instruction_count)

            index_entries.append(entry)

            current_data_offset += instruction_count * instruction_size

        for entry in index_entries:
            f.write(entry)

        for core_id in range(args.tile_num):

            for code in all_core_instrs[core_id]:
                f.write(code.to_bytes(16, 'little'))
    print(f"{out_file} generated successfully.")

def parse_instruction_string(line):

    line = line.strip()

    if '//' in line:
        line = line[:line.find('//')].strip()

    match = INSTRUCTION_REGEX.match(line)

    if not match:
        return None

    instr_data = match.groupdict()

    instr = {
        'instruction': instr_data['instruction'],
        'id': int(instr_data['id']),
        'dep': int(instr_data['dep'])
    }

    fields_str = instr_data['fields']

    for field_match in FIELD_REGEX.finditer(fields_str):

        key = field_match.group('key')
        value_str = field_match.group('value')

        if value_str.lower().startswith("0x"):
            instr[key] = value_str
        else:
            instr[key] = int(value_str)

    return instr

if __name__ == "__main__":
    main()