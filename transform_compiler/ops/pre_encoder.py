from typing import List
from .gule_gemm import matmul_gelu
from .softmax import softmax
from .instruction_base import Instruction
from .gemm import matmul
from .gemm_transpose import matmul_transpose
from .gemm_add import matmul_add
from .Layernorm import LayerNorm

def pre_encoder(
    hidden_dim: int, num_heads: int, seq_len: int,

    input_addr,
    output_addr,

    Wq_addr, Wk_addr, Wv_addr, Wo_addr, gamma_addr, 
    Wi_addr, Weo_addr, gamma1_addr,

    Bq_addr, Bk_addr, Bv_addr, Bo_addr, beta_addr, 
    Bi_addr, Beo_addr, beta1_addr,

    Q_buf, K_buf, V_buf,
    score_buf, context_buf, attention_buf, GeLU_buf,

    re_scale_q, re_scale_k, re_scale_v,
    re_shift_q, re_shift_k, re_shift_v,
    scale_q, scale_k, scale_v,
    shift_q, shift_k, shift_v,

    score_scale, score_shift, score_descale, score_deshift,
    prods_scale, prods_shift, prods_descale, prods_deshift,

    context_descale, context_deshift, context_scale, context_shift,
    linear_scale, linear_shift, linear_rescale, linear_reshift,
    
    # Linear Add (Residual 1)
    linear_add_re_scale, linear_add_re_shift,
    linear_add_res_scale, linear_add_res_shift,

    # LayerNorm 1
    layernorm_deshift, layernorm_descale,
    layernorm_shift, layernorm_scale,

    # LayerNorm 2 (Output Norm) 
    norm_scale, norm_shift, norm_re_scale, norm_re_shift,

    # FFN (Intermediate + Gelu)
    Intermediate_shift, Intermediate_scale, Intermediate_re_shift, Intermediate_re_scale,
    gelu_shift, gelu_scale, gelu_re_shift, gelu_re_scale,

    # FFN Output (Encoder Out + Residual 2)
    encoder_shift, encoder_scale, encoder_re_shift, encoder_re_scale,
    encoder_add_re_shift, encoder_add_res_shift, encoder_add_re_scale, encoder_add_res_scale,

    lbuf_base: int,
    lbuf_stride: int,
    sa_size: int,
    dram_zero: int
) -> List[str]:
    
    instrs: List[str] = []
    instr_id = 0
    head_dim = hidden_dim // num_heads

    w_head_size = hidden_dim * head_dim
    b_head_size = head_dim
    wo_head_size = head_dim * hidden_dim 
    buf_head_size = seq_len * head_dim

    instr_id = LayerNorm(
        input_addr=input_addr, output_addr=attention_buf,
        input_height=seq_len, input_width=hidden_dim,
        de_shift=layernorm_deshift, de_scale=layernorm_descale,
        shift=layernorm_shift, scale=layernorm_scale,
        lbuf_base=lbuf_base, instrs=instrs, instr_id=0,
        layer_name="LayerNorm1_Pre"
    )
    instrs.append(Instruction.End())

    attn_input_addr = attention_buf

    for h in range(num_heads):
        curr_w_offset = h * w_head_size
        curr_b_offset = h * b_head_size
        curr_buf_offset = h * buf_head_size
        
        instr_id = matmul(
            addr_a_base=attn_input_addr, 
            addr_b_base=Wq_addr + curr_w_offset, 
            addr_c_base=Q_buf + curr_buf_offset,
            Width_A=hidden_dim, Height_A=seq_len,
            Width_B=head_dim, Height_B=hidden_dim,
            lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
            sa_size=sa_size, dram_zero=dram_zero,
            re_shift=re_shift_q, re_scale=re_scale_q,
            shift=shift_q, scale=scale_q,
            instrs=instrs, layer_name=f"Q_proj_h{h}",
            bias_base=Bq_addr + curr_b_offset, has_bias=True
        )
        instrs.append(Instruction.End())

        instr_id = matmul(
            addr_a_base=attn_input_addr, 
            addr_b_base=Wk_addr + curr_w_offset, 
            addr_c_base=K_buf + curr_buf_offset,
            Width_A=hidden_dim, Height_A=seq_len,
            Width_B=head_dim, Height_B=hidden_dim,
            lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
            sa_size=sa_size, dram_zero=dram_zero,
            re_shift=re_shift_k, re_scale=re_scale_k,
            shift=shift_k, scale=scale_k,
            instrs=instrs, layer_name=f"K_proj_h{h}",
            bias_base=Bk_addr + curr_b_offset, has_bias=True
        )
        instrs.append(Instruction.End())

        instr_id = matmul(
            addr_a_base=attn_input_addr, 
            addr_b_base=Wv_addr + curr_w_offset, 
            addr_c_base=V_buf + curr_buf_offset,
            Width_A=hidden_dim, Height_A=seq_len,
            Width_B=head_dim, Height_B=hidden_dim,
            lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
            sa_size=sa_size, dram_zero=dram_zero,
            re_shift=re_shift_v, re_scale=re_scale_v,
            shift=shift_v, scale=scale_v,
            instrs=instrs, layer_name=f"V_proj_h{h}",
            bias_base=Bv_addr + curr_b_offset, has_bias=True
        )
        instrs.append(Instruction.End())

        instr_id = matmul_transpose(
            addr_a_base=Q_buf + curr_buf_offset,
            addr_b_base=K_buf + curr_buf_offset,
            addr_c_base=score_buf,
            Width_A=head_dim, Height_A=seq_len,
            Width_B=head_dim, Height_B=seq_len,
            lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
            sa_size=sa_size, dram_zero=dram_zero,
            re_shift=score_deshift, re_scale=score_descale,
            shift=score_shift, scale=score_scale,
            instrs=instrs, layer_name=f"Score_h{h}",
            bias_base=None, has_bias=False
        )
        instrs.append(Instruction.End())

        instr_id = softmax(
            input_addr=score_buf, output_addr=score_buf,
            input_height=seq_len, input_width=seq_len,
            de_shift=prods_deshift, de_scale=prods_descale,
            shift=prods_shift, scale=prods_scale,
            lbuf_base=lbuf_base, instrs=instrs, instr_id=0,
            layer_name=f"Softmax_h{h}"
        )
        instrs.append(Instruction.End())

        instr_id = matmul(
            addr_a_base=score_buf,
            addr_b_base=V_buf + curr_buf_offset,
            addr_c_base=context_buf + curr_buf_offset,
            Width_A=seq_len, Height_A=seq_len,
            Width_B=head_dim, Height_B=seq_len,
            lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
            sa_size=sa_size, dram_zero=dram_zero,
            re_shift=context_deshift, re_scale=context_descale,
            shift=context_shift, scale=context_scale,
            instrs=instrs, layer_name=f"Context_h{h}",
            bias_base=None, has_bias=False
        )
        instrs.append(Instruction.End())

    block1_out_addr = attention_buf

    for h in range(num_heads):
        curr_ctx_offset = h * buf_head_size
        curr_wo_offset = h * wo_head_size
        
        is_first_head = (h == 0)
        addr_add = input_addr if is_first_head else block1_out_addr
        
        curr_bias = Bo_addr if is_first_head else None
        has_bias = True if is_first_head else False

        instr_id = matmul_add(
            addr_a_base=context_buf + curr_ctx_offset, 
            addr_b_base=Wo_addr + curr_wo_offset, 
            addr_c_base=block1_out_addr, 
            addr_add_base=addr_add,
            Width_A=head_dim, Height_A=seq_len,
            Width_B=hidden_dim, Height_B=head_dim,
            lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
            sa_size=sa_size, dram_zero=dram_zero,
            re_shift=linear_reshift, re_scale=linear_rescale,
            shift=linear_shift, scale=linear_scale,
            add_reshift=linear_add_re_shift, add_rescale=linear_add_re_scale,
            add_res_shift=linear_add_res_shift, add_res_scale=linear_add_res_scale,
            instrs=instrs, layer_name=f"Linear_Add_h{h}",
            bias_base=curr_bias, has_bias=has_bias
        )
        instrs.append(Instruction.End())

    instr_id = LayerNorm(
        input_addr=attention_buf, output_addr=context_buf,
        input_height=seq_len, input_width=hidden_dim,
        de_shift=norm_re_shift, de_scale=norm_re_scale,
        shift=norm_shift, scale=norm_scale,
        lbuf_base=lbuf_base, instrs=instrs, instr_id=0,
        layer_name="LayerNorm2_Pre"
    )
    instrs.append(Instruction.End())
    
    ffn_input_addr = context_buf

    instr_id = matmul_gelu(
        addr_a_base=ffn_input_addr, addr_b_base=Wi_addr, addr_c_base=GeLU_buf,
        Width_A=hidden_dim, Height_A=seq_len,
        Width_B=4 * hidden_dim, Height_B=hidden_dim,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=Intermediate_re_shift, re_scale=Intermediate_re_scale,
        shift=Intermediate_shift, scale=Intermediate_scale,
        gelu_re_shift=gelu_re_shift, gelu_re_scale=gelu_re_scale,
        gelu_shift=gelu_shift, gelu_scale=gelu_scale,
        instrs=instrs, layer_name="FFN_GeLU",
        bias_base=Bi_addr, has_bias=True
    )
    instrs.append(Instruction.End())

    
    instr_id = matmul_add(
        addr_a_base=GeLU_buf, addr_b_base=Weo_addr, 
        addr_c_base=output_addr, 
        addr_add_base=attention_buf,
        Width_A=4 * hidden_dim, Height_A=seq_len,
        Width_B=hidden_dim, Height_B=4 * hidden_dim,
        lbuf_base=lbuf_base, lbuf_stride=lbuf_stride,
        sa_size=sa_size, dram_zero=dram_zero,
        re_shift=encoder_re_shift, re_scale=encoder_re_scale,
        shift=encoder_shift, scale=encoder_scale,
        add_reshift=encoder_add_re_shift, add_rescale=encoder_add_re_scale,
        add_res_shift=encoder_add_res_shift, add_res_scale=encoder_add_res_scale,
        instrs=instrs, layer_name="FFN_Out_Add",
        bias_base=Beo_addr, has_bias=True
    )
    instrs.append(Instruction.End())

    return instrs