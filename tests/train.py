import dataclasses
import tempfile

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training import utils as training_utils

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _human_readable_size(num_bytes: float) -> str:
    num_bytes = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def _leaf_info(x) -> str:
    arr = x.value if hasattr(x, "value") else x
    num_params = arr.size
    num_bytes = arr.size * arr.dtype.itemsize
    return f"{arr.shape}@{arr.dtype} | params={num_params:,} | size={_human_readable_size(num_bytes)}"


def _count_params(state: nnx.State, filter: nnx.filterlib.Filter) -> tuple[int, int]:
    """返回 (参数数量, 参数字节数)"""
    filtered_state = state.filter(filter)
    leaves = jax.tree_util.tree_leaves(filtered_state)
    total_params = 0
    total_bytes = 0
    for x in leaves:
        arr = x.value if hasattr(x, "value") else x
        total_params += arr.size
        total_bytes += arr.size * arr.dtype.itemsize
    return total_params, total_bytes


def _shape_bytes(pytree) -> int:
    """对 eval_shape 产出的 shape-only pytree，计算总字节数。"""
    total = 0
    for leaf in jax.tree_util.tree_leaves(pytree):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            total += leaf.size * leaf.dtype.itemsize
    return total


# ---------------------------------------------------------------------------
# 变体定义
# ---------------------------------------------------------------------------

VARIANTS = [
    ("gemma_2b_lora+gemma_300m",      "gemma_2b_lora",  "gemma_300m"),
    ("gemma_2b_lora+gemma_300m_lora", "gemma_2b_lora",  "gemma_300m_lora"),
]


# ---------------------------------------------------------------------------
# 1. 创建模型并分析冻结 / 可训练参数
# ---------------------------------------------------------------------------

def create_and_analyze_variant(variant_name: str, paligemma_variant: str, action_expert_variant: str):
    print(f"\n{'='*80}")
    print(f"=== Variant: {variant_name}  (paligemma={paligemma_variant}, action_expert={action_expert_variant}) ===")
    print(f"{'='*80}")

    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        pcd=False,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )
    model = config.create(key)

    print(f"  model_type: {config.model_type}")
    print(f"  max_token_len: {config.max_token_len}, discrete_state_input: {config.discrete_state_input}")

    state = nnx.state(model)

    total_params, total_bytes = _count_params(state, nnx.All())
    print(f"\n  Total parameters:  {total_params:,}")
    print(f"  Total size:        {_human_readable_size(total_bytes)} ({total_bytes:,} bytes)")

    freeze_filter = config.get_freeze_filter()
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))

    frozen_params, frozen_bytes = _count_params(state, nnx.All(nnx.Param, freeze_filter))
    trainable_params, trainable_bytes = _count_params(state, trainable_filter)

    print(f"\n  --- Freeze Filter: {freeze_filter} ---")
    print(f"  Frozen parameters:    {frozen_params:,}  ({_human_readable_size(frozen_bytes)})")
    print(f"  Trainable parameters: {trainable_params:,}  ({_human_readable_size(trainable_bytes)})")
    print(f"  Trainable ratio:      {trainable_params / total_params * 100:.2f}%")

    print("\n  --- Trainable Params Detail ---")
    trainable_state = state.filter(trainable_filter)
    print(training_utils.tree_to_info(trainable_state, _leaf_info))

    return config, model, total_params, total_bytes, trainable_params, trainable_bytes, freeze_filter, trainable_filter


# ---------------------------------------------------------------------------
# 2. Checkpoint 保存 / 加载测试
#    完全对齐 scripts/train.py 中 init_train_state 的真实逻辑：
#    - 冻结参数转 bfloat16
#    - 优化器仅对可训练参数初始化 (tx.init(trainable_params))
#    - 用 jax.eval_shape 推导 TrainState 形状，不实际分配内存
#    - 用小型 mock TrainState 做实际的 save/load roundtrip 测试
# ---------------------------------------------------------------------------

def test_checkpoint_save_load(config, model, freeze_filter, trainable_filter):
    """测试 checkpoint 的保存（不进行恢复，避免 OOM）。

    对齐 train.py init_train_state 的真实逻辑：
    - 冻结参数转 bfloat16
    - 优化器仅对可训练参数初始化: tx.init(params.filter(trainable_filter))
    - 用 jax.eval_shape 推导真实 TrainState 形状与字节数
    """
    print("\n  === Checkpoint Save Test ===")

    # ── 对齐 train.py init_train_state.init ──
    graphdef, state = nnx.split(model)
    params = state

    # 冻结参数转 bfloat16
    params = nnx_utils.state_map(
        params, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
    )

    trainable_params = params.filter(trainable_filter)

    # 优化器仅对可训练参数初始化
    tx = _optimizer.create_optimizer(_optimizer.AdamW(), _optimizer.CosineDecaySchedule())
    opt_state = tx.init(trainable_params)

    train_state = training_utils.TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=opt_state,
        ema_decay=None,
        ema_params=None,
    )

    # 用 eval_shape 推导形状与字节数（不额外分配显存）
    train_state_shape = jax.eval_shape(lambda: train_state)
    params_bytes = _shape_bytes(train_state_shape.params)
    opt_state_bytes = _shape_bytes(train_state_shape.opt_state)
    print(f"  params shape total:      {_human_readable_size(params_bytes)}")
    print(f"  opt_state shape total:   {_human_readable_size(opt_state_bytes)}")
    print(f"  opt_state / params ratio: {opt_state_bytes / params_bytes:.2f}x")

    # 保存 checkpoint（仅测试保存，不进行恢复）
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = epath.Path(tmpdir)

        mngr = ocp.CheckpointManager(
            checkpoint_dir,
            item_handlers={
                "train_state": ocp.PyTreeCheckpointHandler(),
                "params": ocp.PyTreeCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=3,
                create=True,
                async_options=ocp.AsyncOptions(timeout_secs=60),
            ),
        )

        def _split_params(ts: training_utils.TrainState):
            if ts.ema_params is not None:
                return dataclasses.replace(ts, ema_params=None), ts.ema_params
            return dataclasses.replace(ts, params=nnx.State({})), ts.params

        # 保存 step 0
        ts_split, params_out = _split_params(train_state)
        mngr.save(0, {"train_state": ts_split, "params": {"params": params_out}})
        mngr.wait_until_finished()
        print("  [✓] Saved step 0 (no EMA)")

        # 保存 step 1
        train_state_s1 = dataclasses.replace(train_state, step=jnp.array(1, dtype=jnp.int32))
        ts_split_s1, params_out_s1 = _split_params(train_state_s1)
        mngr.save(1, {"train_state": ts_split_s1, "params": {"params": params_out_s1}})
        mngr.wait_until_finished()
        print("  [✓] Saved step 1")

        # 保存 step 2 (with EMA)
        train_state_s2 = dataclasses.replace(
            train_state,
            step=jnp.array(2, dtype=jnp.int32),
            ema_decay=0.99,
            ema_params=params,
        )
        ts_split_s2, params_out_s2 = _split_params(train_state_s2)
        mngr.save(2, {"train_state": ts_split_s2, "params": {"params": params_out_s2}})
        mngr.wait_until_finished()
        print("  [✓] Saved step 2 (with EMA)")

        available_steps = tuple(mngr.all_steps())
        print(f"  Available steps: {available_steps}")

        mngr.close()
        print("  === Checkpoint Save Test Passed ===\n")

    return params_bytes, opt_state_bytes


# ---------------------------------------------------------------------------
# 3. 前向传播 + 中间激活值追踪 + 显存估算
# ---------------------------------------------------------------------------

def forward_pass(config, model, batch_size, total_params, total_bytes, trainable_params, trainable_bytes):
    print(f"\n  === Forward Pass (batch_size={batch_size}) ===")

    observation = config.fake_obs(batch_size=batch_size)
    actions = config.fake_act(batch_size=batch_size)

    print("  Observation shapes:")
    for name, img in observation.images.items():
        print(f"    image[{name}]: {img.shape}@{img.dtype}")
    print(f"    state: {observation.state.shape}@{observation.state.dtype}")
    print(f"    tokenized_prompt: {observation.tokenized_prompt.shape}@{observation.tokenized_prompt.dtype}")
    print(f"    actions: {actions.shape}@{actions.dtype}")

    rng = jax.random.key(1)
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

    act_mem_details = {}
    total_act_bytes = 0

    observation_p = _model.preprocess_observation(preprocess_rng, observation, train=False)
    print("  After preprocess_observation:")
    for name, img in observation_p.images.items():
        print(f"    image[{name}]: {img.shape}")

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    for name, arr in [("noise", noise), ("time", time), ("x_t", x_t), ("u_t", u_t)]:
        act_mem_details[name] = arr.nbytes
        total_act_bytes += arr.nbytes
    print(f"  noise: {noise.shape}, time: {time.shape}, x_t: {x_t.shape}, u_t: {u_t.shape}")

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation_p)
    for name, arr in [("prefix_tokens", prefix_tokens), ("prefix_mask", prefix_mask), ("prefix_ar_mask", prefix_ar_mask)]:
        act_mem_details[name] = arr.nbytes
        total_act_bytes += arr.nbytes
    print(f"  prefix_tokens: {prefix_tokens.shape}, prefix_mask: {prefix_mask.shape}")

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation_p, x_t, time)
    for name, arr in [("suffix_tokens", suffix_tokens), ("suffix_mask", suffix_mask), ("suffix_ar_mask", suffix_ar_mask)]:
        act_mem_details[name] = arr.nbytes
        total_act_bytes += arr.nbytes
    if adarms_cond is not None:
        act_mem_details["adarms_cond"] = adarms_cond.nbytes
        total_act_bytes += adarms_cond.nbytes
    print(f"  suffix_tokens: {suffix_tokens.shape}, adarms_cond: {None if adarms_cond is None else adarms_cond.shape}")

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    for name, arr in [("input_mask", input_mask), ("ar_mask", ar_mask), ("attn_mask", attn_mask), ("positions", positions)]:
        act_mem_details[name] = arr.nbytes
        total_act_bytes += arr.nbytes
    print(f"  attn_mask: {attn_mask.shape}, positions: {positions.shape}")

    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    if prefix_out is not None:
        act_mem_details["prefix_out"] = prefix_out.nbytes
        total_act_bytes += prefix_out.nbytes
    act_mem_details["suffix_out"] = suffix_out.nbytes
    total_act_bytes += suffix_out.nbytes
    print(f"  prefix_out: {None if prefix_out is None else prefix_out.shape}, suffix_out: {suffix_out.shape}")

    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    act_mem_details["v_t"] = v_t.nbytes
    total_act_bytes += v_t.nbytes
    print(f"  v_t: {v_t.shape}")

    loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
    act_mem_details["loss"] = loss.nbytes
    total_act_bytes += loss.nbytes
    print(f"  per-step loss shape: {loss.shape}, mean: {jnp.mean(loss)}")

    print("\n  === Intermediate Activations Memory Breakdown ===")
    for name, bytes_ in act_mem_details.items():
        print(f"    {name}: {_human_readable_size(bytes_)} ({bytes_:,} bytes)")
    print(f"    TOTAL ACTIVATIONS: {_human_readable_size(total_act_bytes)} ({total_act_bytes:,} bytes)")

    return total_act_bytes


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH_SIZE = 2
    results_summary = []

    for variant_name, paligemma_v, action_expert_v in VARIANTS:
        (config, model, total_params, total_bytes,
         trainable_params, trainable_bytes,
         freeze_filter, trainable_filter) = create_and_analyze_variant(
            variant_name, paligemma_v, action_expert_v
        )

        # Checkpoint 测试：返回真实 TrainState 中 params 和 opt_state 的字节数
        params_bytes, opt_state_bytes = test_checkpoint_save_load(
            config, model, freeze_filter, trainable_filter
        )

        # 前向传播 + 激活值
        total_act_bytes = forward_pass(
            config, model, BATCH_SIZE,
            total_params, total_bytes,
            trainable_params, trainable_bytes,
        )

        # ── 真实 GPU 显存估算（对齐 train.py 的初始化逻辑）──
        # 冻结参数 → bfloat16 (2 bytes)，可训练参数 → float32 (4 bytes)
        # 梯度：仅可训练参数，float32
        # 优化器：仅可训练参数的 Adam m + v，float32
        grad_bytes = trainable_bytes
        # opt_state_bytes 已由 eval_shape 精确计算
        total_est = params_bytes + grad_bytes + opt_state_bytes + total_act_bytes

        print("\n  === GPU Memory Breakdown Estimate (aligned with train.py init_train_state) ===")
        print(f"    Parameters (mixed dtype):  {_human_readable_size(params_bytes)} ({params_bytes:,} bytes)")
        print(f"      ├─ Frozen (bfloat16):     {_human_readable_size(params_bytes - trainable_bytes)}")
        print(f"      └─ Trainable (float32):   {_human_readable_size(trainable_bytes)}")
        print(f"    Gradients (trainable f32):  {_human_readable_size(grad_bytes)}")
        print(f"    Optimizer (Adam, trainable):{_human_readable_size(opt_state_bytes)} ({opt_state_bytes:,} bytes)")
        print(f"    Activations (est.):         {_human_readable_size(total_act_bytes)}")
        print(f"    {'─'*55}")
        print(f"    TOTAL ESTIMATED:            {_human_readable_size(total_est)} ({total_est:,} bytes)")

        results_summary.append({
            "variant": variant_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "trainable_ratio": f"{trainable_params / total_params * 100:.2f}%",
            "params_bytes": params_bytes,
            "opt_state_bytes": opt_state_bytes,
            "activation_bytes": total_act_bytes,
            "total_gpu_est": total_est,
        })

        # 清理
        del model, config
        jax.clear_caches()
        import gc
        gc.collect()

    # ── 汇总对比表 ──
    print(f"\n{'='*110}")
    print(f"=== SUMMARY: Variant Comparison (batch_size={BATCH_SIZE}) ===")
    print(f"{'='*110}")
    print(f"{'Variant':<35} {'Total Params':>14} {'Trainable':>14} {'Frozen':>14} {'Ratio':>8} {'Params(B)':>10} {'OptState(B)':>12} {'GPU Est':>12}")
    print(f"{'─'*110}")
    for r in results_summary:
        print(f"{r['variant']:<35} {r['total_params']:>14,} {r['trainable_params']:>14,} {r['frozen_params']:>14,} {r['trainable_ratio']:>8} {_human_readable_size(r['params_bytes']):>10} {_human_readable_size(r['opt_state_bytes']):>12} {_human_readable_size(r['total_gpu_est']):>12}")
