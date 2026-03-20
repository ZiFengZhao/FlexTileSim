import argparse
import csv
import math
import re
import os
import struct
from typing import Dict, Any, List

from gemm import gen_inst
from softmax import gen_softmax_inst

ADDR_BITS = 32
GDMA_T = 0x01
CE_T = 0x02
CU_T = 0x03
MVIN_T = 0x00
MVOUT_T = 0x01
GEMM_T = 0x00
RET_T = 0x01
SOFTMAX_T = 0x02
LAYERNORM_T = 0x03
ADD_T = 0x04
GELU_T = 0x05
XPOSE_T = 0x06
RELU_T = 0x07
SYNC_T = 0x00
NOP_T = 0x01

FIELDS = {
    'OPCODE':           [0, 8],
    'FUNCT0':           [8, 8],
    'INST_ID':          [16, 8],
    'DEP_TYPE':         [24, 2],
    'GDMA_STRIDE':      [26, 15],
    'GDMA_IMM0_HEIGHT': [41, 15],
    'GDMA_IMM1_WIDTH':  [56, 8],
    'ADDR0':            [64, ADDR_BITS],
    'ADDR1':            [96, ADDR_BITS],
    'CE_IMM0':          [26, 8],
    'CE_IMM1':          [34, 8],
    'CE_IMM2':          [42, 8],
    'CE_IMM3':          [50, 8],
    'CE_FUNCT1':        [58, 6],
    'SYNC_CORE_COUNT':  [64, ADDR_BITS],
    'SYNC_ID':          [96, ADDR_BITS],
}
ENCODERS = {}

def encode_field(machine_code: int, field_name: str, value: int) -> int:
    start, width = FIELDS[field_name]
    mask = (1 << width) - 1
    machine_code &= ~((mask) << start)
    machine_code |= (value & mask) << start
    return machine_code

def encode_gdma(instr: Dict[str, Any], inst_type: int) -> int:
    machine_code = 0
    machine_code = encode_field(machine_code, 'OPCODE', GDMA_T)
    machine_code = encode_field(machine_code, 'FUNCT0', inst_type)
    machine_code = encode_field(machine_code, 'INST_ID', instr['id'])
    machine_code = encode_field(machine_code, 'DEP_TYPE', instr['dep'])
    machine_code = encode_field(machine_code, 'GDMA_STRIDE', instr['stride'])
    machine_code = encode_field(machine_code, 'GDMA_IMM0_HEIGHT', instr['Height'])
    machine_code = encode_field(machine_code, 'GDMA_IMM1_WIDTH', instr['Width'])
    
    src_addr = int(instr['src'],16) if isinstance(instr['src'], str) else instr['src']
    dst_addr = int(instr['dst'],16) if isinstance(instr['dst'], str) else instr['dst']
    addr0 = src_addr if inst_type == MVOUT_T else dst_addr
    addr1 = dst_addr if inst_type == MVOUT_T else src_addr
    machine_code = encode_field(machine_code, 'ADDR0', addr0)
    machine_code = encode_field(machine_code, 'ADDR1', addr1)
    return machine_code

def encode_ce(instr: Dict[str, Any], inst_type: int) -> int:
    machine_code = 0
    machine_code = encode_field(machine_code, 'OPCODE', CE_T)
    machine_code = encode_field(machine_code, 'FUNCT0', inst_type)
    machine_code = encode_field(machine_code, 'INST_ID', instr['id'])
    machine_code = encode_field(machine_code, 'DEP_TYPE', instr['dep'])

    imm0, imm1, imm2, imm3, funct1 = 0, 0, 0, 0, 0
    addr0_str, addr1_str = "0x0", "0x0"

    if inst_type == GEMM_T:
        addr0_str = instr['src0']
        addr1_str = instr['src1']
    elif inst_type == RET_T:
        imm0, imm1, imm2, imm3 = instr['scale'], instr['shift'], instr['re_scale'], instr['re_shift']
        funct1 = instr['bias']
        addr0_str = instr['bias_addr']
        addr1_str = instr['ocm_addr']
    elif inst_type == ADD_T:
        imm0, imm1, imm2, imm3 = instr['res_scale'], instr['res_shift'], instr['re_scale'], instr['re_shift']
        addr0_str = instr['res']
        addr1_str = instr['main']
    elif inst_type in [SOFTMAX_T, LAYERNORM_T, GELU_T]:
        imm0, imm1, imm2, imm3 = instr['scale'], instr['shift'], instr['de_scale'], instr['de_shift']
        addr0_str = instr['ocm_in']
        addr1_str = instr['ocm_out']
    elif inst_type in [XPOSE_T, RELU_T]:
        addr0_str = instr['ocm_in']
        addr1_str = instr['ocm_out']
    else:
        raise ValueError(f"Unknown CE instruction type: {inst_type}")

    machine_code = encode_field(machine_code, 'CE_IMM0', imm0)
    machine_code = encode_field(machine_code, 'CE_IMM1', imm1)
    machine_code = encode_field(machine_code, 'CE_IMM2', imm2)
    machine_code = encode_field(machine_code, 'CE_IMM3', imm3)
    machine_code = encode_field(machine_code, 'CE_FUNCT1', funct1)
    
    # *** 修正点 ***: 直接用 int(str, 16) 转换, 不再需要复杂的正则表达式
    addr0 = int(addr0_str, 16) if isinstance(addr0_str, str) else addr0_str
    addr1 = int(addr1_str, 16) if isinstance(addr1_str, str) else addr1_str
    
    machine_code = encode_field(machine_code, 'ADDR0', addr0)
    machine_code = encode_field(machine_code, 'ADDR1', addr1)
    return machine_code


def encode_cu(instr: Dict[str, Any], inst_type: int) -> int:
    machine_code = 0
    machine_code = encode_field(machine_code, 'OPCODE', CU_T)
    machine_code = encode_field(machine_code, 'FUNCT0', inst_type)
    machine_code = encode_field(machine_code, 'INST_ID', instr['id'])
    
    addr0 = instr['sync_core_count'] if inst_type == SYNC_T else 0
    addr1 = instr['sync_id'] if inst_type == SYNC_T else 0
    
    machine_code = encode_field(machine_code, 'ADDR0', addr0)
    machine_code = encode_field(machine_code, 'ADDR1', addr1)
    return machine_code


ENCODERS['MVIN'] = lambda i: encode_gdma(i, MVIN_T)
ENCODERS['MVOUT'] = lambda i: encode_gdma(i, MVOUT_T)
ENCODERS['GEMM'] = lambda i: encode_ce(i, GEMM_T)
ENCODERS['RET'] = lambda i: encode_ce(i, RET_T)
ENCODERS['ADD'] = lambda i: encode_ce(i, ADD_T)
ENCODERS['SOFTMAX'] = lambda i: encode_ce(i, SOFTMAX_T)
ENCODERS['LAYERNORM'] = lambda i: encode_ce(i, LAYERNORM_T)
ENCODERS['GELU'] = lambda i: encode_ce(i, GELU_T)
ENCODERS['TRANSPOSE'] = lambda i: encode_ce(i, XPOSE_T)
ENCODERS['RELU'] = lambda i: encode_ce(i, RELU_T)
ENCODERS['SYNC'] = lambda i: encode_cu(i, SYNC_T)
ENCODERS['NOP'] = lambda i: encode_cu(i, NOP_T)

INSTRUCTION_REGEX = re.compile(
    r'^\s*(?P<instruction>[A-Za-z]+)\s+'
    r'(?P<id>\d+)\s+'
    r'dep=(?P<dep>\d+)\s*'
    r'(?P<fields>.*)'
)
FIELD_REGEX = re.compile(r'(?P<key>[A-Za-z0-9_]+)=(?P<value>[xX0-9A-Fa-f]+)')
CORE_HEADER_REGEX = re.compile(r'^#\s*Core\s+(?P<core_id>\d+)\s+instructions')


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ScaleSim CSV to GEMM m/n/k for multi-tile simulator")
    parser.add_argument("--model", type=str, required=True, help="Target DNN model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to ScaleSim CSV topology file")
    parser.add_argument("--tile_num", type=int, default=1, help="Number of tiles (for layer splitting)")
    parser.add_argument("--sa_size", type=int, default=32, help="Size of systolic array")
    parser.add_argument("--binary_only", "-b", action="store_true", help="Only output binary file")
    return parser.parse_args()

def transform_conv(layer):
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

    ops = []

    ops.append((layer_name, "GEMM", m, n, k))

    return ops

def transform_transformer(layer):

    name = layer['Layer name']
    L = int(layer['Seq Length'])
    D = int(layer['Embed Dim'])
    H = int(layer['Num Heads'])
    F = int(layer['FFN Dim'])

    ops = []

    ops.append((name+"_QKV", "GEMM", L, 3*D, D))
    ops.append((name+"_AttnScore", "GEMM", L, L, D))
    ops.append((name+"_Softmax", "SOFTMAX", L, L, None))
    ops.append((name+"_Context", "GEMM", L, D, L))
    ops.append((name+"_OutProj", "GEMM", L, D, D))

    return ops

def detect_model_type(header):

    if "IFMAP Height" in header:
        return "CNN"

    if "Seq Length" in header:
        return "TRANSFORMER"

    raise ValueError("Unknown CSV format")

def split_layer(layer_name, m, n, k, tile_num):
    base_m = m // tile_num
    remainder = m % tile_num
    result = []
    for i in range(tile_num):
        tile_m = base_m + (1 if i < remainder else 0)
        tile_name = layer_name
        result.append((tile_name, tile_m, n, k))
    return result

def get_ddr_channels(num_cores):

    if num_cores in [1,2]:
        return 1
    if num_cores in [4,8]:
        return 2
    if num_cores == 16:
        return 4
    if num_cores in [32,64]:
        return 8

    raise ValueError("Invalid number of cores")

def parse_instr_str(instr_str):
    import re
    INSTRUCTION_REGEX = re.compile(
        r'^\s*(?P<instruction>[A-Za-z]+)\s+'
        r'(?P<id>\d+)\s+dep=(?P<dep>\d+)\s*(?P<fields>.*)'
    )
    FIELD_REGEX = re.compile(r'(?P<key>[A-Za-z0-9_]+)=(?P<value>[xX0-9A-Fa-f]+)')
    
    match = INSTRUCTION_REGEX.match(instr_str)
    if not match:
        raise ValueError(f"Cannot parse instruction string: {instr_str}")
    instr_data = match.groupdict()
    
    instr_dict = {
        'instruction': instr_data['instruction'].upper(),
        'id': int(instr_data['id']),
        'dep': int(instr_data['dep'])
    }
    
    for field_match in FIELD_REGEX.finditer(instr_data['fields']):
        key = field_match.group('key')
        val_str = field_match.group('value')
        instr_dict[key] = int(val_str, 0) if val_str.startswith('0x') else int(val_str)
    
    return instr_dict

def asm_to_bin(core_instructions: dict, output_filepath: str):
    index_entry_format = '<QI'  # offset (uint64), count (uint32)
    index_entry_size = struct.calcsize(index_entry_format)
    instruction_size = 16

    max_core_id = max(core_instructions.keys())

    data_area_start_offset = (max_core_id + 1) * index_entry_size

    with open(output_filepath, 'wb') as f:
        current_data_offset = data_area_start_offset
        index_entries = []
        for i in range(max_core_id + 1):
            if i in core_instructions:
                count = len(core_instructions[i])
                offset = current_data_offset
                current_data_offset += count * instruction_size
            else:
                count = 0
                offset = 0
            index_entries.append(struct.pack(index_entry_format, offset, count))

        for entry in index_entries:
            f.write(entry)

        for i in range(max_core_id + 1):
            if i not in core_instructions:
                continue
            for instr in core_instructions[i]:
                instr_name = instr['instruction'].upper()
                code = ENCODERS[instr_name](instr)
                f.write(code.to_bytes(instruction_size, 'little'))

def main():
    args = parse_args()

    num_cores = args.tile_num
    num_ddr_channels = get_ddr_channels(args.tile_num)

    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        model_type = detect_model_type(header)
        layers = [{k.strip(): v.strip() for k,v in row.items()} for row in reader]

    all_core_instrs = {i:[] for i in range(args.tile_num)}
    prev_layer_instr_id = [0 for _ in range(args.tile_num)]

    for layer in layers:
        if model_type == "CNN":
            ops = transform_conv(layer)
        else:
            ops = transform_transformer(layer)
        
        for op_name, op_type, m, n, k in ops:
            if op_type == "GEMM":
                mnk_per_tile = split_layer(op_name,m,n,k,args.tile_num)

                for tile_id,(_,tile_m,n,k) in enumerate(mnk_per_tile):
                    instrs = gen_inst(
                        M=tile_m,
                        N=n,
                        K=k,
                        sa_size=args.sa_size,
                        core_id=tile_id,
                        num_cores=args.tile_num,
                        num_ddr_channels=num_ddr_channels,
                        instr_id=prev_layer_instr_id[tile_id]
                    )
                    
                    all_core_instrs[tile_id].extend([parse_instr_str(inst) for inst in instrs])
                    prev_layer_instr_id[tile_id]+=len(instrs)

            elif op_type=="SOFTMAX":
                for tile_id in range(args.tile_num):
                    instrs = gen_softmax_inst(
                        core_id=tile_id,
                        seq_size=m*n,
                        instr_id=prev_layer_instr_id[tile_id]
                    )
                    all_core_instrs[tile_id].extend([parse_instr_str(inst) for inst in instrs])
                    prev_layer_instr_id[tile_id]+=len(instrs)

        if args.tile_num > 1:
            for tile_id in range(args.tile_num):
                current_inst_id = prev_layer_instr_id[tile_id]
                sync_inst = f"Sync {len(instrs)} dep=00 sync_id={current_inst_id} sync_core_count={args.tile_num}"
                all_core_instrs[tile_id].append(parse_instr_str(sync_inst))
                prev_layer_instr_id[tile_id] = current_inst_id + 1
        

    out_inst_file = args.model + "_"+ (str)(args.tile_num) + "tile" + "_inst.txt"
    
    if not args.binary_only:
        with open(out_inst_file, "w") as f:
            for core_id in range(args.tile_num):
                f.write(f"# Core {core_id} instructions\n")
                for instr in all_core_instrs[core_id]:
                    fields = ' '.join([
                        f"{k}={hex(v) if isinstance(v, int) and ( 'addr' in k.lower() or k.lower() in ('src','dst') ) else v}" 
                        for k, v in instr.items() 
                        if k not in ('instruction', 'id', 'dep')
                    ])
                    line = f"{instr['instruction']} {instr['id']} dep={instr['dep']} {fields}"
                    f.write(line + "\n")
                f.write("\n")
        
        print(f"{out_inst_file} generated")

    out_bin_file = args.model + "_"+ (str)(args.tile_num) + "tile_inst.bin"
    asm_to_bin(all_core_instrs, out_bin_file)
    print(f"{out_bin_file} generated")

if __name__ == "__main__":
    main()