import ops

def twos_complement(val, bits=8):
    if val < 0:
        val = (1 << bits) + val
    return format(val & ((1 << bits) - 1), f'0{bits}b')




if __name__ == "__main__":
    print(type(twos_complement(-6,8)))
    """
    instrs = []
    softmax.softmax(input_addr=DRAM_BASE_A, input_height=129, input_width=128, de_shift=4, de_scale=2, shift=250, scale=127, 
            instrs=instrs, instr_id=0, layer_name="")
    for i in instrs:
        print(i)
    """
    

