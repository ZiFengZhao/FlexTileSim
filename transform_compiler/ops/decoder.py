from typing import List
from .gule_gemm import matmul_gelu
from .softmax import softmax
from .instruction_base import Instruction
from .gemm import matmul
from .gemm_transpose import matmul_transpose
from .gemm_add import matmul_add
from .Layernorm import LayerNorm

def decoder(
    hidden_dim: int, num_heads: int, seq_len: int,

    input_addr,
    output_addr,

    Wq_addr, Wk_addr, Wv_addr, Wo_addr, Wi_addr, Weo_addr,
    Bq_addr, Bk_addr, Bv_addr, Bo_addr, Bi_addr, Beo_addr,
    Q_buf, K_buf, V_buf,
    score_buf: int,
    context_buf: int,
    GeLU_buf: int,
    layernorm_buf: int,

    re_scale_q, re_scale_k, re_scale_v,
    re_shift_q, re_shift_k, re_shift_v,
    scale_q, scale_k, scale_v,
    shift_q, shift_k, shift_v,

    score_scale: int, score_shift: int, score_descale: int, score_deshift: int,
  
    prods_scale: int, prods_shift: int, prods_descale: int, prods_deshift: int,

    context_descale: int, context_deshift: int, context_scale: int, context_shift: int,

    linear_scale: int, linear_shift: int, linear_rescale: int, linear_reshift: int,
    linear_add_re_scale: int, linear_add_re_shift: int, linear_add_res_scale: int, linear_add_res_shift: int,

    layernorm_deshift: int, layernorm_descale: int,
    layernorm_shift: int, layernorm_scale: int,

    Intermediate_shift: int, Intermediate_scale: int, Intermediate_re_shift: int, Intermediate_re_scale: int,
    gelu_shift: int, gelu_scale: int, gelu_re_shift: int, gelu_re_scale: int,

    encoder_shift: int, encoder_scale: int, encoder_re_shift: int, encoder_re_scale: int,
    encoder_add_re_shift: int, encoder_add_res_shift: int, encoder_add_re_scale: int, encoder_add_res_scale: int,

    lbuf_base: int,
    lbuf_stride: int,
    sa_size: int,
    dram_zero: int
) -> List[str]:
    instrs: List[str] = []
    instr_id = 0

    head_dim = hidden_dim // num_heads

    instr_id = LayerNorm(
        input_addr=input_addr, output_addr=layernorm_buf,
        input_height=seq_len, input_width=hidden_dim,
        de_shift=layernorm_deshift, de_scale=layernorm_descale,
        shift=layernorm_shift, scale=layernorm_scale,
        lbuf_base=lbuf_base, instrs=instrs,
        instr_id=0,
        layer_name="LayerNorm"
    )
    instrs.append(Instruction.End())

    instr_id = matmul(
        addr_a_base=layernorm_buf, addr_b_base=Wq_addr, addr_c_base=Q_buf,
        Width_A=hidden_dim, Height_A=seq_len,
        Width_B=num_heads * head_dim, Height_B=hidden_dim,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=re_shift_q, re_scale=re_scale_q,
        shift=shift_q, scale=scale_q,
        instrs=instrs, layer_name="Q_proj",
        bias_base=Bq_addr, has_bias=True
    )
    instrs.append(Instruction.End())

    instr_id = matmul(
        addr_a_base=layernorm_buf, addr_b_base=Wk_addr, addr_c_base=K_buf,
        Width_A=hidden_dim, Height_A=seq_len,
        Width_B=num_heads * head_dim, Height_B=hidden_dim,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=re_shift_k, re_scale=re_scale_k,
        shift=shift_k, scale=scale_k,
        instrs=instrs, layer_name="K_proj",
        bias_base=Bk_addr, has_bias=True
    )
    instrs.append(Instruction.End())

    instr_id = matmul(
        addr_a_base=layernorm_buf, addr_b_base=Wv_addr, addr_c_base=V_buf,
        Width_A=hidden_dim, Height_A=seq_len,
        Width_B=num_heads * head_dim, Height_B=hidden_dim,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=re_shift_v, re_scale=re_scale_v,
        shift=shift_v, scale=scale_v,
        instrs=instrs, layer_name="V_proj",
        bias_base=Bv_addr, has_bias=True
    )
    instrs.append(Instruction.End())

    instr_id = matmul_transpose(
        addr_a_base=Q_buf,
        addr_b_base=K_buf,
        addr_c_base=score_buf,
        Width_A=head_dim * num_heads, Height_A=seq_len,
        Width_B=head_dim * num_heads, Height_B=seq_len,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=score_deshift, re_scale=score_descale,
        shift=score_shift, scale=score_scale,
        instrs=instrs, layer_name="score",
        bias_base=None, has_bias=False
    )
    instrs.append(Instruction.End())

    prod_buf = score_buf 
    instr_id = softmax(
        input_addr=score_buf,
        output_addr=prod_buf,
        input_height=seq_len,
        input_width=seq_len,
        de_shift=prods_deshift,
        de_scale=prods_descale,
        shift=prods_shift,
        scale=prods_scale,
        lbuf_base=lbuf_base,
        instrs=instrs,
        instr_id=0,
        layer_name="Softmax"
    )
    instrs.append(Instruction.End())

    instr_id = matmul(
        addr_a_base=prod_buf,
        addr_b_base=V_buf,
        addr_c_base=context_buf,
        Width_A=seq_len, Height_A=seq_len,
        Width_B=num_heads * head_dim, Height_B=seq_len,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=context_deshift, re_scale=context_descale,
        shift=context_shift, scale=context_scale,
        instrs=instrs, layer_name="Context",
        bias_base=None, has_bias=False
    )
    instrs.append(Instruction.End())

    linear_out = Q_buf
    instr_id = matmul_add(
        addr_a_base=context_buf, addr_b_base=Wo_addr, addr_c_base=linear_out, addr_add_base=input_addr,
        Width_A=num_heads * head_dim, Height_A=seq_len,
        Width_B=hidden_dim, Height_B=num_heads * head_dim,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=linear_reshift, re_scale=linear_rescale,
        shift=linear_shift, scale=linear_scale,
        add_reshift=linear_add_re_shift, add_rescale=linear_add_re_scale,
        add_res_shift=linear_add_res_shift, add_res_scale=linear_add_res_scale,
        instrs=instrs, layer_name="Add Linear",
        bias_base=Bo_addr, has_bias=True
    )
    instrs.append(Instruction.End())

    layernorm_out1 = K_buf
    instr_id = LayerNorm(
        input_addr=linear_out, output_addr=layernorm_out1,
        input_height=seq_len, input_width=hidden_dim,
        de_shift=layernorm_deshift, de_scale=layernorm_descale,
        shift=layernorm_shift, scale=layernorm_scale,
        lbuf_base=lbuf_base, instrs=instrs,
        instr_id=0,
        layer_name="LayerNorm1"
    )
    instrs.append(Instruction.End())

    GeLU_out = GeLU_buf
    instr_id = matmul_gelu(
        addr_a_base=layernorm_out1, addr_b_base=Wi_addr, addr_c_base=GeLU_out,
        Width_A=hidden_dim, Height_A=seq_len,
        Width_B=4 * hidden_dim, Height_B=hidden_dim,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=Intermediate_re_shift, re_scale=Intermediate_re_scale,
        shift=Intermediate_shift, scale=Intermediate_scale,
        gelu_re_shift=gelu_re_shift, gelu_re_scale=gelu_re_scale,
        gelu_shift=gelu_shift, gelu_scale=gelu_scale,
        instrs=instrs, layer_name="GeLU",
        bias_base=Bi_addr, has_bias=True
    )
    instrs.append(Instruction.End())

    encoder_out = output_addr
    instr_id = matmul_add(
        addr_a_base=GeLU_out, addr_b_base=Weo_addr, addr_c_base=encoder_out, addr_add_base=linear_out,
        Width_A=4 * hidden_dim, Height_A=seq_len,
        Width_B=hidden_dim, Height_B=4 * hidden_dim,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=encoder_re_shift, re_scale=encoder_re_scale,
        shift=encoder_shift, scale=encoder_scale,
        add_reshift=encoder_add_re_shift, add_rescale=encoder_add_re_scale,
        add_res_shift=encoder_add_res_shift, add_res_scale=encoder_add_res_scale,
        instrs=instrs, layer_name="Add Output",
        bias_base=Beo_addr, has_bias=True
    )
    instrs.append(Instruction.End())

    return instrs
