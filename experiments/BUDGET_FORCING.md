# Budget Forcing:强制拉长 think 输出

## 背景

BAGEL think 模式的自然输出长度只有 ~160 token(见 `think_cap_outputs/trials.csv`,cap=256/1000 时 0% 截断)。要研究更长 think(如 1000 token)下的延迟/质量,需要强制模型不提前停止。

采用 s1 论文(simple test-time scaling)的 **budget forcing** 方案:解码到 `min_length` 之前,模型每次想输出 EOS(`<|im_end|>`),都把它替换成一句转折插入语(默认 `" Wait,"`)重新喂入,诱导模型继续思考;过了 `min_length` 后 EOS 正常放行。

## 改动

| 文件 | 改动 |
|------|------|
| `modeling/bagel/bagel.py` | `generate_text` 新增 `min_length`、`wait_token_ids` 参数;token 选择后检测 EOS,`step < min_length` 时用 `forced_queue` 把多 token 插入语逐个注入(注入期间覆盖模型自身预测) |
| `inferencer.py` | `gen_text` 新增 `min_length`、`wait_interjection`(字符串,内部 tokenize 后下传);`interleave_inference` 新增 `min_think_token_n`、`think_wait_interjection` 透传 |
| `experiments/run_cap_sweep_mp.py` | 顶部新增 `FORCE_THINK_LENGTH` / `WAIT_INTERJECTION` 开关;开启后每个 condition 的 `min_think_token_n = max_think_token_n`(min = max = cap,think 长度精确钉死在 cap);两个字段随 `**cond` 写入 trials.csv |

所有新参数都有默认值(`min_length=0` / `wait_interjection=None`),不开启时行为与原版逐 token 一致。

## 怎么跑

```bash
# 1. 打开开关
#    experiments/run_cap_sweep_mp.py: FORCE_THINK_LENGTH = True

# 2. 正常启动并行 sweep
python experiments/run_cap_sweep_mp.py --gpus 0,1,2,3,4,5,6,7
```

单独调用(不走 sweep 脚本):

```python
text = inferencer.gen_text(
    gen_context, do_sample=False, temperature=0.3,
    max_length=1000, min_length=1000, wait_interjection=" Wait,",
)
```

预期:cap=1000 档 think 真正跑满 ~1000 token,t_think ≈ 0.055 × 1000 ≈ 55 s;N=50 时 ratio ≈ 0.104 × 1000/50 ≈ 2.1,think 首次全面压过 image。

## 注意事项

- **贪心解码 + 强制续写会出现重复内容**("车轱辘话")。对延迟测量无影响(解码耗时只与 token 数有关);若关心内容连贯性,把 `run_trial` 的 `do_sample=True` 并配合适中 temperature。
- 强制批次中 `hit_cap` 按构造必为 True,`think_closed` 可能为 False(模型写完 `</think>` 后被迫继续)。这两列在强制批次中含义已变,分析时用 `min_think_token_n > 0` 区分强制/自然两批数据。
- `min_length` 语义是"至少",注入队列排空前不会 break,实际长度可能比 min 多出插入语的几个 token;min = max 时由 `max_length` 截断,长度精确等于 cap。
