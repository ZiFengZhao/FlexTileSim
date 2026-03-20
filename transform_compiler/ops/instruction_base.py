"""
指令基类
"""
from abc import ABC

class Instruction(ABC):
    def Mvin(ddr_addr, ocm_addr, Width_tensor, Height_tensor, stride, dep, instr_id, comment):
        return (
            f"Mvin {instr_id} dep={dep} "
            f"src=0x{ddr_addr:08x} dst=0x{ocm_addr:08x} "
            f"Width={Width_tensor} Height={Height_tensor} stride={stride} // {comment}"
        )

    def Mvout(ddr_addr, ocm_addr, Width_tensor, Height_tensor, stride, dep, instr_id, comment):
        return (
            f"Mvout {instr_id} dep={dep} "
            f"dst=0x{ddr_addr:08x} src=0x{ocm_addr:08x} "
            f"Width={Width_tensor} Height={Height_tensor} stride={stride} // {comment}"
        )

    def GEMM(ocm_addr_1, ocm_addr_2, dep, instr_id, comment):
        return (
            f"GEMM {instr_id} dep={dep} "
            f"src0=0x{ocm_addr_1:08x} src1=0x{ocm_addr_2:08x} // {comment}"
        )

    def RET(ocm_addr, bias_addr, bias, re_shift, re_scale, shift, scale, dep, instr_id, comment):
        return (
            f"RET {instr_id} dep={dep} "
            f"ocm_addr=0x{ocm_addr:08x} bias_addr=0x{bias_addr:08x} bias={bias} "
            f"re_shift={re_shift} re_scale={re_scale} shift={shift} scale={scale} // {comment}"
        )

    def Transpose(ocm_addr_in, ocm_addr_out, dep, instr_id, comment):
        return (
            f"Transpose {instr_id} dep={dep} "
            f"ocm_in=0x{ocm_addr_in:08x} ocm_out=0x{ocm_addr_out:08x} // {comment}"
        )
        
    def Softmax(addr_out, addr_in, de_shift, de_scale, shift, scale, dep, instr_id, comment):
        return (
            f"Softmax {instr_id} dep={dep} "
            f"ocm_out=0x{addr_out:08x} ocm_in=0x{addr_in:08x} "
            f"de_shift={de_shift} de_scale={de_scale} "
            f"shift={shift} scale={scale} // {comment}"
        )

    def LayerNorm(addr_out, addr_in, de_shift, de_scale, shift, scale, dep, instr_id, comment):
        return (
            f"Layernorm {instr_id} dep={dep} "
            f"ocm_out=0x{addr_out:08x} ocm_in=0x{addr_in:08x} "
            f"de_shift={de_shift} de_scale={de_scale} "
            f"shift={shift} scale={scale} // {comment}"
        )

    def Add(addr_main, addr_res, re_shift, re_scale, res_shift, res_scale, dep, instr_id, comment):
        return (
            f"Add {instr_id} dep={dep} "
            f"main=0x{addr_main:08x} res=0x{addr_res:08x} "
            f"re_shift={re_shift} re_scale={re_scale} "
            f"res_shift={res_shift} res_scale={res_scale} // {comment}"
        )
    def Gelu(ocm_addr_out, ocm_addr_in, de_shift, de_scale, shift, scale, dep, instr_id, comment):
        return(
            f"Gelu {instr_id} dep={dep} "
            f"ocm_out=0x{ocm_addr_out:08x} ocm_in=0x{ocm_addr_in:08x} "
            f"de_shift={de_shift} de_scale={de_scale} "
            f"shift={shift} scale={scale} // {comment}"
        )
    def Relu(ocm_addr_out, ocm_addr_in,  dep, instr_id, comment):
        return (
            f"Relu {instr_id} dep={dep} "
            f"ocm_out=0x{ocm_addr_out:08x} ocm_in=0x{ocm_addr_in:08x} // {comment}"
        )
    def End():
        return (f"End "
                f"\n")