#!/usr/bin/env python3
# hw_probe.py - RTX 5060 硬件档案实测（P0 roofline 地基）
#
# 实测项：
#   1) 设备基本信息：名称 / 计算能力 / SM 数 / 显存 / L2
#   2) 显存带宽：device-to-device 大块 copy（读+写 = 2×size/time）
#   3) PCIe 带宽：pinned host <-> device，H2D 与 D2H 分开
#   4) PCIe 链路：nvidia-smi 查询世代/通道（理论峰值参考）
#
# 输出：quant/hardware_profile.yaml，并打印结论摘要
#
# 注意：torch CUDA 初始化需要访问 /proc，若在受限沙箱内运行会报
#       Error 304，需在沙箱外执行。

import argparse
import statistics
import subprocess
import time

import torch
import yaml


def pci_link_info():
    """通过 nvidia-smi 查 PCIe 链路世代/通道（可能返回 None）。"""
    query = "pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        gmax, gcur, wmax, wcur = [x.strip() for x in out.split(",")]
        return {
            "gen_max": int(gmax), "gen_current": int(gcur),
            "width_max": int(wmax), "width_current": int(wcur),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def benchmark_copy(src, dst, iters=20, warmup=5):
    """对 dst.copy_(src) 计时，返回中位数耗时(ms)。"""
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        dst.copy_(src)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def measure():
    props = torch.cuda.get_device_properties(0)
    mib = 1024 ** 2
    result = {
        "device": {
            "name": str(props.name),
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": int(props.multi_processor_count),
            "total_memory_mib": round(props.total_memory / mib, 1),
            "l2_cache_kib": int(props.L2_cache_size // 1024),
            "memory_bus_width_bit": int(props.memory_bus_width),
            "memory_clock_mhz": int(props.memory_clock_rate // 1000),
            "max_threads_per_sm": int(props.max_threads_per_multi_processor),
            "warp_size": int(props.warp_size),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": str(torch.__version__),
    }

    # 1 GiB 测试块：8GB 卡上 src+dst = 2GB 安全；耗时 ~5ms 精度足够
    size = gib = 1024 ** 3
    result["test_block_gib"] = 1

    print(f"[1/3] 显存带宽（D2D, 1 GiB block）...")
    src = torch.empty(size, dtype=torch.uint8, device="cuda")
    dst = torch.empty(size, dtype=torch.uint8, device="cuda")
    src.fill_(0xAB)
    ms = benchmark_copy(src, dst)
    result["vram"] = {
        "median_ms": round(ms, 4),
        "bandwidth_gib_s": round(2 * gib / mib / ms, 1),   # 读 + 写
        "bandwidth_gb_s": round(2 * size / 1e9 / ms * 1e3, 1),
    }
    del src, dst

    print("[2/3] PCIe H2D（pinned -> device）...")
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    host.fill_(0xCD)
    dst = torch.empty(size, dtype=torch.uint8, device="cuda")
    ms_h2d = benchmark_copy(host, dst)

    print("[3/3] PCIe D2H（device -> pinned）...")
    ms_d2h = benchmark_copy(dst, host)
    result["pcie"] = {
        "h2d": {
            "median_ms": round(ms_h2d, 4),
            "bandwidth_gib_s": round(gib / mib / ms_h2d, 1),
            "bandwidth_gb_s": round(size / 1e9 / ms_h2d * 1e3, 1),
        },
        "d2h": {
            "median_ms": round(ms_d2h, 4),
            "bandwidth_gib_s": round(gib / mib / ms_d2h, 1),
            "bandwidth_gb_s": round(size / 1e9 / ms_d2h * 1e3, 1),
        },
    }
    del host, dst

    result["pcie_link"] = pci_link_info()

    # 推导式锚点（P0 报告用）：MXFP4 0.96GB/token 在实测 H2D 带宽下的上限
    h2d_gbs = result["pcie"]["h2d"]["bandwidth_gb_s"]
    result["derived"] = {
        "mxfp4_zero_cache_tok_s_ceiling": round(h2d_gbs / 0.96, 1),
        "note": "每 token 0.96GB / 实测 H2D 有效带宽；未含非专家访存与计算，为宽松上界",
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="quant/hardware_profile.yaml")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA 不可用（若 Error 304 请在沙箱外运行）"
    result = measure()

    with open(args.output, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)

    d, v, p = result["device"], result["vram"], result["pcie"]
    link = result["pcie_link"]
    print("\n" + "=" * 60)
    print(" HARDWARE PROFILE 结论")
    print("=" * 60)
    print(f" GPU:            {d['name']} (sm_{d['compute_capability']})")
    print(f" SM 数:          {d['sm_count']}")
    print(f" 显存:            {d['total_memory_mib']} MiB")
    print(f" L2:             {d['l2_cache_kib']} KiB")
    print(f" 显存带宽实测:    {v['bandwidth_gb_s']} GB/s（标称 448）")
    print(f" PCIe H2D 实测:  {p['h2d']['bandwidth_gb_s']} GB/s")
    print(f" PCIe D2H 实测:  {p['d2h']['bandwidth_gb_s']} GB/s")
    if "error" not in link:
        print(f" PCIe 链路:      Gen{link['gen_current']} x{link['width_current']}"
              f"（max Gen{link['gen_max']} x{link['width_max']}）")
    print(f" MXFP4 零缓存上界: {result['derived']['mxfp4_zero_cache_tok_s_ceiling']} tok/s")
    print("=" * 60)
    print(f" 已落盘: {args.output}")


if __name__ == "__main__":
    main()
