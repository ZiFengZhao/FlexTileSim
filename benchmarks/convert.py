# --- Usage (用法) ---
#
# 运行脚本:
# python3 convert.py -i <输入汇编文件>
#
# 示例:
# python3 convert.py -i example_gemm_dual_core.txt
#
# 脚本将生成一个二进制文件:
# example_gemm_dual_core.bin
#
# --------------------

import re
import argparse
import os
import struct
from typing import Dict, Any, List

# --- 1. 指令集配置  ---
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

# --- 2. 核心编码函数 ---

def encode_field(machine_code: int, field_name: str, value: int) -> int:
    """将值编码到机器码的指定字段位置"""
    start, width = FIELDS[field_name]
    mask = (1 << width) - 1
    machine_code &= ~((mask) << start)
    machine_code |= (value & mask) << start
    return machine_code

def encode_gdma(instr: Dict[str, Any], inst_type: int) -> int:
    """将GDMA指令编码为机器码"""
    machine_code = 0
    machine_code = encode_field(machine_code, 'OPCODE', GDMA_T)
    machine_code = encode_field(machine_code, 'FUNCT0', inst_type)
    machine_code = encode_field(machine_code, 'INST_ID', instr['id'])
    machine_code = encode_field(machine_code, 'DEP_TYPE', instr['dep'])
    machine_code = encode_field(machine_code, 'GDMA_STRIDE', instr['stride'])
    machine_code = encode_field(machine_code, 'GDMA_IMM0_HEIGHT', instr['Height'])
    machine_code = encode_field(machine_code, 'GDMA_IMM1_WIDTH', instr['Width'])
    
    src_addr = int(instr['src'], 16)
    dst_addr = int(instr['dst'], 16)
    addr0 = src_addr if inst_type == MVOUT_T else dst_addr
    addr1 = dst_addr if inst_type == MVOUT_T else src_addr
    machine_code = encode_field(machine_code, 'ADDR0', addr0)
    machine_code = encode_field(machine_code, 'ADDR1', addr1)
    return machine_code

def encode_ce(instr: Dict[str, Any], inst_type: int) -> int:
    """将CE指令编码为机器码"""
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
    addr0 = int(addr0_str, 16)
    addr1 = int(addr1_str, 16)
    
    machine_code = encode_field(machine_code, 'ADDR0', addr0)
    machine_code = encode_field(machine_code, 'ADDR1', addr1)
    return machine_code


def encode_cu(instr: Dict[str, Any], inst_type: int) -> int:
    """将CU指令编码为机器码"""
    machine_code = 0
    machine_code = encode_field(machine_code, 'OPCODE', CU_T)
    machine_code = encode_field(machine_code, 'FUNCT0', inst_type)
    machine_code = encode_field(machine_code, 'INST_ID', instr['id'])
    
    addr0 = instr['sync_core_count'] if inst_type == SYNC_T else 0
    addr1 = instr['sync_id'] if inst_type == SYNC_T else 0
    
    machine_code = encode_field(machine_code, 'ADDR0', addr0)
    machine_code = encode_field(machine_code, 'ADDR1', addr1)
    return machine_code


# --- 3. 编码函数映射字典填充 ---
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

# --- 4. 解析器实现 ---
INSTRUCTION_REGEX = re.compile(
    r'^\s*(?P<instruction>[A-Za-z]+)\s+'
    r'(?P<id>\d+)\s+'
    r'dep=(?P<dep>\d+)\s*'
    r'(?P<fields>.*)'
)
FIELD_REGEX = re.compile(r'(?P<key>[A-Za-z0-9_]+)=(?P<value>[xX0-9A-Fa-f]+)')
CORE_HEADER_REGEX = re.compile(r'^#\s*Core\s+(?P<core_id>\d+)\s+instructions')

def parse_assembly_file(filepath: str) -> Dict[int, List[Dict[str, Any]]]:
    """解析汇编文件"""
    core_instructions: Dict[int, List[Dict[str, Any]]] = {}
    current_core_id = -1
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '//' in line:
                line = line[:line.find('//')].strip()

            core_match = CORE_HEADER_REGEX.match(line)
            if core_match:
                current_core_id = int(core_match.group('core_id'))
                if current_core_id not in core_instructions:
                    core_instructions[current_core_id] = []
                continue
            
            if current_core_id == -1 or not line or line.startswith('#'):
                continue
            
            match = INSTRUCTION_REGEX.match(line)
            if match:
                instr_data = match.groupdict()
                instr: Dict[str, Any] = {
                    'instruction': instr_data['instruction'],
                    'id': int(instr_data['id']),
                    'dep': int(instr_data['dep']),
                }
                fields_str = instr_data['fields']
                for field_match in FIELD_REGEX.finditer(fields_str):
                    key = field_match.group('key')
                    value_str = field_match.group('value')
                    instr[key] = int(value_str) if not value_str.lower().startswith('0x') else value_str
                
                core_instructions[current_core_id].append(instr)
    return core_instructions


def main():
    parser = argparse.ArgumentParser(description="NPU Assembly to Binary Converter.")
    parser.add_argument("--input_file", "-i", type=str, help="Input assembly file path.", required=True)
    args = parser.parse_args()
    
    input_filepath = args.input_file
    base, _ = os.path.splitext(input_filepath)
    output_filepath = base + ".bin"
    
    # 1. 解析汇编文件
    core_instructions_parsed = parse_assembly_file(input_filepath)
    if not core_instructions_parsed:
        print(f"Warning: No instructions found in {input_filepath}.")
        raise SystemExit

    # 2. 将所有指令编码为 128-bit 整数
    core_machine_codes: Dict[int, List[int]] = {}
    for core_id, instructions in core_instructions_parsed.items():
        codes = []
        for instr in instructions:
            instr_name = instr['instruction'].upper()
            machine_code = ENCODERS[instr_name](instr)
            codes.append(machine_code)
        core_machine_codes[core_id] = codes

    # 3. 准备写入二进制文件
    # C++ 结构体: uint64_t offset (8 bytes), uint32_t count (4 bytes)
    # Python struct format: '<QI' (Little-endian, unsigned long long, unsigned int)
    index_entry_format = '<QI'
    index_entry_size = struct.calcsize(index_entry_format)
    instruction_size = 16 # 128 bits = 16 bytes

    max_core_id = max(core_machine_codes.keys()) if core_machine_codes else -1
    
    # 数据区的起始位置在所有索引项之后
    data_area_start_offset = (max_core_id + 1) * index_entry_size
    
    print(f"Found cores up to ID {max_core_id}. Index table size: {data_area_start_offset} bytes.")
    print(f"Writing binary output to {output_filepath}...")

    with open(output_filepath, 'wb') as f:
        # --- 4. 写入索引区 ---
        index_entries = []
        current_data_offset = data_area_start_offset
        
        for i in range(max_core_id + 1):
            if i in core_machine_codes:
                instruction_count = len(core_machine_codes[i])
                offset = current_data_offset
                
                # 创建当前 Core 的索引项
                entry = struct.pack(index_entry_format, offset, instruction_count)
                index_entries.append(entry)
                
                # 更新下一个 Core 数据区的起始偏移
                current_data_offset += instruction_count * instruction_size
            else:
                # 如果这个 Core ID 不存在, 写入一个空的索引项
                entry = struct.pack(index_entry_format, 0, 0)
                index_entries.append(entry)
        
        # 将所有索引项一次性写入文件
        for entry in index_entries:
            f.write(entry)

        # --- 5. 写入数据区 ---
        for i in range(max_core_id + 1):
            if i in core_machine_codes:
                codes = core_machine_codes[i]
                for code in codes:
                    # 将 128-bit 整数转换为 16 字节的 bytes 对象 (little-endian)
                    f.write(code.to_bytes(instruction_size, 'little'))

    print("Binary file generation complete.")


if __name__ == "__main__":
    main()
