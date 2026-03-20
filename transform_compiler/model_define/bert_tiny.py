import os
import sys
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ops.encoder import encoder

DRAM_ZERO      = 0xF000_0000  
LBUF_BASE      = 0x0000_1020
LBUF_STRIDE    = 0x0000_1000
DRAM_BASE_A    = 0x0000_9020  
DRAM_BASE_B    = 0x2000_0000  
DRAM_BASE_BIAS = 0x4000_0000  
DRAM_BASE_C    = 0x6000_0000  


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


def make_layer_addresses(layer_idx: int) -> dict:
    weight_stride = 0x0010_0000
    bias_stride = 0x0008_0000

    layer_w_base = DRAM_BASE_B + layer_idx * weight_stride
    layer_b_base = DRAM_BASE_BIAS + layer_idx * bias_stride

    return dict(
        Wq=layer_w_base + 0x00000,
        Wk=layer_w_base + 0x04000,
        Wv=layer_w_base + 0x08000,
        Wo=layer_w_base + 0x0C000,
        gamma=layer_w_base + 0x10000,
        Wi=layer_w_base + 0x14000,
        Weo=layer_w_base + 0x24000,
        gamma1=layer_w_base + 0x34000,
        Bq=layer_b_base + 0x00000,
        Bk=layer_b_base + 0x04000,
        Bv=layer_b_base + 0x08000,
        Bo=layer_b_base + 0x0C000,
        beta=layer_b_base + 0x10000,
        Bi=layer_b_base + 0x14000,
        Beo=layer_b_base + 0x24000,
        beta1=layer_b_base + 0x34000,
    )


def main():
    num_layers = 2
    hidden_dim = 128
    num_heads = 2
    seq_len = 128

    act_ping = DRAM_BASE_C + 0x40000
    act_pong = DRAM_BASE_C + 0x50000

    Q_buf = DRAM_BASE_C + 0x00000
    K_buf = DRAM_BASE_C + 0x02000
    V_buf = DRAM_BASE_C + 0x04000
    score_buf = DRAM_BASE_C + 0x06000
    context_buf = DRAM_BASE_C + 0x08000
    attention_buf = DRAM_BASE_C + 0x0A000
    GeLU_buf = DRAM_BASE_C + 0x12000

    quant_params = build_quant_params()

    instrs: List[str] = []

    for layer_idx in range(num_layers):
        layer_in = DRAM_BASE_A if layer_idx == 0 else (act_ping if (layer_idx - 1) % 2 == 0 else act_pong)
        layer_out = act_ping if layer_idx % 2 == 0 else act_pong

        addr = make_layer_addresses(layer_idx)

        instrs.extend(
            encoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                seq_len=seq_len,
                input_addr=layer_in,
                output_addr=layer_out,
                Wq_addr=addr["Wq"], Wk_addr=addr["Wk"], Wv_addr=addr["Wv"],
                Wo_addr=addr["Wo"], gamma_addr=addr["gamma"],
                Wi_addr=addr["Wi"], Weo_addr=addr["Weo"], gamma1_addr=addr["gamma1"],
                Bq_addr=addr["Bq"], Bk_addr=addr["Bk"], Bv_addr=addr["Bv"],
                Bo_addr=addr["Bo"], beta_addr=addr["beta"],
                Bi_addr=addr["Bi"], Beo_addr=addr["Beo"], beta1_addr=addr["beta1"],
                Q_buf=Q_buf, K_buf=K_buf, V_buf=V_buf,
                score_buf=score_buf, context_buf=context_buf,
                attention_buf=attention_buf, GeLU_buf=GeLU_buf,
                lbuf_base=LBUF_BASE, lbuf_stride=LBUF_STRIDE,
                sa_size=32, dram_zero=DRAM_ZERO,
                **quant_params,
            )
        )

    final_output = act_pong if num_layers % 2 == 0 else act_ping
    print(f"// BERT-Tiny instructions generated. Final output addr: 0x{final_output:X}")
    for inst in instrs:
        print(inst)


if __name__ == "__main__":
    main()