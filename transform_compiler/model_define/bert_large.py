import os
import sys
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from ops.encoder import encoder

DRAM_ZERO       = 0xF000_0000
LBUF_BASE       = 0x0000_1020
LBUF_STRIDE     = 0x0000_1000
DRAM_BASE_A     = 0x0000_9020
DRAM_BASE_B     = 0x2000_0000
DRAM_BASE_C     = 0x6000_0000
DRAM_BASE_BIAS  = 0x4000_0000

def build_quant_params() -> dict:
    return dict(
        re_scale_q=4, re_scale_k=4, re_scale_v=4,
        re_shift_q=4, re_shift_k=4, re_shift_v=4,
        scale_q=2, scale_k=2, scale_v=2,
        shift_q=36, shift_k=36, shift_v=36,
        score_scale=2, score_shift=36, score_descale=4, score_deshift=4,
        prods_scale=127, prods_shift=-6, prods_descale=2, prods_deshift=4,
        context_descale=4, context_deshift=4, context_scale=2, context_shift=36,
        linear_scale=2, linear_shift=36, linear_rescale=4, linear_reshift=4,
        linear_add_re_scale=4, linear_add_re_shift=4,
        linear_add_res_scale=2, linear_add_res_shift=4,
        layernorm_deshift=16, layernorm_descale=11,
        layernorm_shift=-4, layernorm_scale=131,
        norm_scale=2, norm_shift=36, norm_re_scale=4, norm_re_shift=4,
        Intermediate_shift=36, Intermediate_scale=2,
        Intermediate_re_shift=4, Intermediate_re_scale=4,
        gelu_shift=36, gelu_scale=2, gelu_re_shift=4, gelu_re_scale=4,
        encoder_shift=36, encoder_scale=2,
        encoder_re_shift=4, encoder_re_scale=4,
        encoder_add_re_shift=4, encoder_add_res_shift=4,
        encoder_add_re_scale=4, encoder_add_res_scale=2,
    )

def main():
    num_layers = 24
    hidden_dim = 1024
    num_heads  = 16
    seq_len    = 128

    SZ_PROJ = 0x100000   # 1024 x 1024
    SZ_FFN  = 0x400000   # 1024 x 4096
    SZ_VEC  = 0x2000

    LAYER_WEIGHT_STRIDE = 0x1400000  
    LAYER_BIAS_STRIDE   = 0x0200000  

    ping_addr = DRAM_BASE_C
    pong_addr = DRAM_BASE_C + 0x200000
    internal_buf_base = DRAM_BASE_C + 0x500000

    Q_buf         = internal_buf_base
    K_buf         = internal_buf_base + 0x080000
    V_buf         = internal_buf_base + 0x100000
    score_buf     = internal_buf_base + 0x180000
    context_buf   = internal_buf_base + 0x280000
    attention_buf = internal_buf_base + 0x380000
    GeLU_buf      = internal_buf_base + 0x480000

    quant_params = build_quant_params()
    instrs: List[str] = []

    for i in range(num_layers):
        layer_in = DRAM_BASE_A if i == 0 else (ping_addr if (i % 2 == 1) else pong_addr)
        layer_out = ping_addr if (i % 2 == 0) else pong_addr

        w_base = DRAM_BASE_B + i * LAYER_WEIGHT_STRIDE
        b_base = DRAM_BASE_BIAS + i * LAYER_BIAS_STRIDE

        Wq_addr = w_base
        Wk_addr = Wq_addr + SZ_PROJ
        Wv_addr = Wk_addr + SZ_PROJ
        Wo_addr = Wv_addr + SZ_PROJ
        gamma_addr = Wo_addr + SZ_PROJ

        Wi_addr = gamma_addr + SZ_VEC
        Weo_addr = Wi_addr + SZ_FFN
        gamma1_addr = Weo_addr + SZ_FFN

        Bq_addr = b_base
        Bk_addr = Bq_addr + SZ_VEC
        Bv_addr = Bk_addr + SZ_VEC
        Bo_addr = Bv_addr + SZ_VEC
        beta_addr = Bo_addr + SZ_VEC

        Bi_addr = beta_addr + SZ_VEC
        Beo_addr = Bi_addr + SZ_FFN
        beta1_addr = Beo_addr + SZ_PROJ

        instrs.extend(
            encoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                seq_len=seq_len,
                input_addr=layer_in,
                output_addr=layer_out,
                Wq_addr=Wq_addr, Wk_addr=Wk_addr, Wv_addr=Wv_addr, Wo_addr=Wo_addr, gamma_addr=gamma_addr,
                Wi_addr=Wi_addr, Weo_addr=Weo_addr, gamma1_addr=gamma1_addr,
                Bq_addr=Bq_addr, Bk_addr=Bk_addr, Bv_addr=Bv_addr, Bo_addr=Bo_addr, beta_addr=beta_addr,
                Bi_addr=Bi_addr, Beo_addr=Beo_addr, beta1_addr=beta1_addr,
                Q_buf=Q_buf, K_buf=K_buf, V_buf=V_buf,
                score_buf=score_buf, context_buf=context_buf, attention_buf=attention_buf, GeLU_buf=GeLU_buf,
                lbuf_base=LBUF_BASE, lbuf_stride=LBUF_STRIDE,
                sa_size=32, dram_zero=DRAM_ZERO,
                **quant_params,
            )
        )

    final_output = pong_addr if (num_layers % 2 == 0) else ping_addr
    print(f"// BERT-Large instructions generated. Final output addr: 0x{final_output:X}")
    for inst in instrs:
        print(inst)

if __name__ == "__main__":
    main()