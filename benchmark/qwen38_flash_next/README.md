# 在 A3 上复现 Qwen3.8-Flash-Next 的双 PR 性能

用一台 A3 的 8 个空闲 die 跑一个副本。历史基线的三轮整批耗时为 **126.80、126.53、127.91 秒**。取后两轮均值是 **127.22 秒**，平均首 token **17.68 秒**，平均每个后续 token **25.36 ms**，输出吞吐 **1030.3 token/s**。

## 1. 装好两个 PR

使用已加载 CANN 9.2.0 环境的 SGLang 镜像。参考环境用 Python 3.12、torch / torch-npu 2.13.0 和 Triton 3.6 系列。框架用本分支（基于 `012aa4d034`），算子仓用 `176434f0`。两个终端使用同一 Python 环境。[SGLang #52107](https://github.com/sgl-project/sglang/pull/52107) · [sgl-kernel-npu #1076](https://github.com/sgl-project/sgl-kernel-npu/pull/1076)

在干净的算子仓根目录编译并安装 wheel。保留两个构建选项，第二个会装上基线使用的 MoE 分组矩阵乘。

```bash
python3 setup.py --arch-version 2201 --third-party ops_transformer,ops_transformer_gmm93
python3 -m pip install --no-deps --force-reinstall output/sgl_kernel_npu-*.whl
```

然后在本分支的 SGLang 仓库根目录运行 `python3 -m pip install --no-deps -e python`。

## 2. 起服务

在 SGLang 仓库根目录运行。把权重目录和设备号换成本机的实际值。

```bash
export MODEL_PATH=/path/to/Qwen3.8-Flash-Next
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash benchmark/qwen38_flash_next/serve.sh
```

服务使用 TP8 / EP8、BF16、NEXTN 三步、PLE 常驻显存。Prefill 每块 4096 token，专家通信用 AllToAllV，decode 用 MC2。KV 池是 469888 token，前缀缓存关闭。等服务显示已就绪，再测速。

## 3. 测速

在同一台机器的第二个终端，从 SGLang 仓库根目录运行。端口默认 30000。需要换端口时，在两个终端设置相同的 `PORT`。

```bash
bash benchmark/qwen38_flash_next/bench.sh
```

脚本使用随附的 32 条合成请求。30 条输入在 8000–11000 token 之间，另外两条分别是 21000 和 30000 token。先用每条输出 16、256 token 的两批请求预热，再测三轮。每个正式轮次先预热 8 条，然后以 32 并发跑完全部请求，每条输出 4096 token。保留同一个服务进程，比较脚本末尾的第 2、3 轮均值。

一轮应成功完成 **32 条、336000 个输入 token、131072 个输出 token**。整批耗时看 `Benchmark duration (s)`。输入文件 SHA-256 为 `88999a151b8c84dbcb67fe2d19532e68fcce7f90b037ec3c499fb3ba71e29a09`。

**如果首 token 明显变慢，先看服务日志。** 基线平均 TTFT 为 **17.68 秒**。32 条长输入分成 83 个 prefill 块，开头有请求排队很正常，热轮最后一条在 36–38 秒才拿到首 token。随后 decode 日志能同时看到 `#running-req: 32` 和 `#queue-req: 0`。如果进入 decode 后队列长时间不清空、同时运行的请求数一直达不到 32，先核对服务实际拿到的请求槽位和 KV 容量。如果能跑满 32 条，再看处理完 83 个 prefill 块花了多久。

**PLE offload 的收益取决于并发。** 固定 32 槽和同样大的 KV 池，开启 offload 前后，平均 TTFT 为 **17.68 / 17.74 秒**。在另一组 64 并发测试中，开启 offload 后把 KV 池扩大到 939776 token、运行槽位从 32 增至 64，平均 TTFT 才从 **79.54 秒降到 35.06 秒**。复现本页的 32 并发基线时，保持 PLE 常驻即可。
