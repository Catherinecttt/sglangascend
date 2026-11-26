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

## 4. 单机 16 die 文本配置

`serve_tp16.sh` 启动一个 TP16 / EP16 副本。Attention 分两组（`--attn-dp-size 2`），每组 8 个 die、32 个请求槽、469888 个 KV token。Prefill 分块参数传 8192，服务端按组折算为 4096。PLE 使用 pinned 主存，MoE 在 prefill 用 AllGather，在 decode 用 MC2。

16 die 实测用了较新的框架和当时的 QSA、MoE 等改动。沿用第 1 节的算子 wheel，在交接分支的仓库根目录执行一次以下命令，把实测代码准备到相邻目录。`origin` 指提供本交接分支的 fork。

```bash
git fetch origin dev/qwen38-flash-next-a3
git worktree add --detach ../sglang-tp16 32390471f7800109e9b9872f952508a8d3ed424f
git -C ../sglang-tp16 apply "$PWD/benchmark/qwen38_flash_next/tp16-runtime.patch"
```

设置好第 2 节的 `MODEL_PATH`，在交接分支的仓库根目录启动服务。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
PYTHONPATH="$PWD/../sglang-tp16/python${PYTHONPATH:+:$PYTHONPATH}" bash benchmark/qwen38_flash_next/serve_tp16.sh
```

原来的 `bench.sh` 可以直接测 32 并发。历史上，同一服务处理随附的 32 条文本、每条输出 4096 token，单轮耗时 **94.47 秒**。将这组输入重复两份，以 64 并发发送 64 条请求，后两轮平均 **123.32 秒**，输出吞吐 **2125.83 token/s**。

## 5. 固定的纯文本和图文混合输入

`serving_benchmark_v1.zip` 包含 A、B、C、D 四组互不重叠的合成请求。每组可取 32 到 256 条，步长为 32。每批一半纯文本、一半带图片，平均输入约 10,720 token，最长 29,990 token，每条固定输出 4,096 token。

在仓库根目录运行以下命令。脚本会校验 ZIP 并展开全部 32 份 JSONL。用 IDE 打开 `serving_benchmark_v1/A032.jsonl` 或 `D032.jsonl`，每行是一条完整请求，图片路径相对于 JSONL 所在目录。

```bash
cd benchmark/qwen38_flash_next
python3 unpack_mixed.py
```

测速时，使用支持图文的服务，上下文至少设为 34,086 token，并安装 `aiohttp`。将 `BASE_URL` 设为以 `/v1` 结尾的 API 地址，`MODEL` 设为服务暴露的模型名。在刚才的目录运行以下命令。

```bash
python3 serving_benchmark_v1/bench.py --base-url "$BASE_URL" --model "$MODEL" --batch A032 --out result-A032.json
```

跨机器比较时，选择同一批次，关闭前缀缓存和图片 embedding 缓存，用独立请求预热。客户端会逐条核对服务端返回的输入和输出 token 数。
