# ── _diag_spec.py — 解析 gpt_model.bin 头部配置 ──
# 格式 (V3): magic 4B | version 4B | precision 1B | model_type 4B
#   GPT: vocab u64, d_model u64, seq_len u64, num_heads u64, d_ff u64, num_layers u64
#        pos_encoding u32, activation u32, norm_type u32
import struct, sys

def main(path="gpt_model.bin"):
    with open(path, "rb") as f:
        head = f.read(96)
    magic, version, precision, mtype = struct.unpack_from("<IIBI", head, 0)
    print(f"magic={magic:#x} version={version} precision={precision}(0=f32,1=f64) model_type={mtype}")
    if mtype == 3 or mtype == 4:  # GPT / ALiBi_GPT
        vs, dm, sl, nh, df, nl = struct.unpack_from("<QQQQQQ", head, 13)
        pe, act, nt = struct.unpack_from("<III", head, 61)
        names = ["Learned", "Sinusoidal", "ALiBi"]
        print(f"  vocab_size = {vs}")
        print(f"  d_model    = {dm}")
        print(f"  seq_len    = {sl}")
        print(f"  num_heads  = {nh}")
        print(f"  d_ff       = {df}")
        print(f"  num_layers = {nl}")
        print(f"  pos_enc    = {names[pe] if pe < 3 else pe}")
        print(f"  activation = {act}")
        print(f"  norm_type  = {nt}")

if __name__ == "__main__":
    main(*sys.argv[1:])
