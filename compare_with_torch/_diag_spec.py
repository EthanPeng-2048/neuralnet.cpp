# ── _diag_spec.py — 解析模型文件头部配置 ──
# 格式 (v4 自描述): magic 4B | version 4B | precision 1B |
#   spec_len u64 | spec KeyValueRecord | weights...
# KeyValueRecord: field_count u32 | field{key_len u32, key, type u8, value_len u32, value}...
#   type: 0=UInt(u64), 1=Str, 2=UIntArray(count×u64)
import struct, sys


def read_kv(raw: bytes):
    """解析 KeyValueRecord 字节 → dict（key → 值）。"""
    pos = 0
    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        return v
    def u64():
        nonlocal pos
        v = struct.unpack_from("<Q", raw, pos)[0]
        pos += 8
        return v

    count = u32()
    out = {}
    for _ in range(count):
        key_len = u32()
        key = raw[pos:pos + key_len].decode()
        pos += key_len
        typ = raw[pos]
        pos += 1
        value_len = u32()
        value = raw[pos:pos + value_len]
        pos += value_len
        if typ == 0:
            out[key] = struct.unpack("<Q", value)[0]
        elif typ == 1:
            out[key] = value.decode(errors="replace")
        elif typ == 2:
            out[key] = [struct.unpack_from("<Q", value, i)[0]
                        for i in range(0, len(value), 8)]
    return out


def main(path="mnist_model.bin"):
    with open(path, "rb") as f:
        head = f.read(96)
    magic, version, precision = struct.unpack_from("<IIB", head, 0)
    spec_len = struct.unpack_from("<Q", head, 9)[0]
    with open(path, "rb") as f:
        f.seek(17)
        spec = read_kv(f.read(spec_len))

    print(f"magic={magic:#x} version={version} precision={precision}(0=f32,1=f64) spec_len={spec_len}")
    names = {0: "Unknown", 1: "MLP", 2: "Transformer", 3: "GPT", 4: "ALiBi_GPT"}
    norm = {0: "LayerNorm", 1: "RMSNorm", 2: "BatchNorm"}
    pos_enc = {0: "Learned", 1: "Sinusoidal", 2: "ALiBi"}
    act = {0: "GeLU", 1: "SwiGLU"}
    mtype = spec.get("type", 0)
    print(f"  type       = {mtype} ({names.get(mtype, '?')})")
    if "layer_dims" in spec:
        print(f"  layer_dims = {spec['layer_dims']}")
    for k in ("d_model", "num_heads", "d_ff", "num_layers", "patch_size",
              "vocab_size", "seq_len"):
        if k in spec:
            print(f"  {k:<11}= {spec[k]}")
    if "pos_encoding" in spec:
        print(f"  pos_enc    = {pos_enc.get(spec['pos_encoding'], spec['pos_encoding'])}")
    if "activation" in spec:
        print(f"  activation = {act.get(spec['activation'], spec['activation'])}")
    if "norm_type" in spec:
        print(f"  norm_type  = {norm.get(spec['norm_type'], spec['norm_type'])}")


if __name__ == "__main__":
    main(*sys.argv[1:])
