from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import utils as training_utils


def _human_readable_size(num_bytes: float) -> str:
    # 适配 int 和 float
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


def create_pi05():
    print("\n=== Create PI0.5 Model ===")
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, action_horizon=32, pcd=False)
    model = config.create(key)

    print(f"model_type: {config.model_type}")
    print(f"max_token_len: {config.max_token_len}, discrete_state_input: {config.discrete_state_input}")

    print("\n=== PI0.5 Model: shape, dtype, params, size ===")
    state = nnx.state(model)
    print(training_utils.tree_to_info(state, _leaf_info))

    leaves = jax.tree_util.tree_leaves(state)
    total_params = 0
    total_bytes = 0
    for x in leaves:
        arr = x.value if hasattr(x, "value") else x
        total_params += arr.size
        total_bytes += arr.size * arr.dtype.itemsize

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size: {_human_readable_size(total_bytes)} ({total_bytes} bytes)")

    # 返回总字节数，用于后续显存估算
    return config, model, total_params, total_bytes


def forward_pass(config, model, batch_size, total_params, total_bytes):
    # 1. 构造 fake 输入
    observation = config.fake_obs(batch_size=batch_size)
    actions = config.fake_act(batch_size=batch_size)

    print("\n=== Observation shapes ===")
    for name, img in observation.images.items():
        print(f"  image[{name}]: {img.shape}@{img.dtype}")
    print(f"  state: {observation.state.shape}@{observation.state.dtype}")
    print(f"  tokenized_prompt: {observation.tokenized_prompt.shape}@{observation.tokenized_prompt.dtype}")
    print(f"  tokenized_prompt_mask: {observation.tokenized_prompt_mask.shape}")
    print(f"  actions: {actions.shape}@{actions.dtype}")

    # 2. 手动复现前向传播，并追踪中间激活张量大小
    rng = jax.random.key(1)
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

    # --- 用于累计激活值显存的字典和计数器 ---
    act_mem_details = {}
    total_act_bytes = 0

    # 记录输入（通常不计入激活，但为了完整可选择性计入，这里不计入，只记录后续中间层）
    observation_p = _model.preprocess_observation(preprocess_rng, observation, train=False)
    print("\n=== After preprocess_observation ===")
    for name, img in observation_p.images.items():
        print(f"  image[{name}]: {img.shape}")

    # (1) 扩散噪声相关
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    # 累计激活：noise, time, x_t, u_t
    act_mem_details['noise'] = noise.nbytes
    act_mem_details['time'] = time.nbytes
    act_mem_details['x_t'] = x_t.nbytes
    act_mem_details['u_t'] = u_t.nbytes
    total_act_bytes += noise.nbytes + time.nbytes + x_t.nbytes + u_t.nbytes

    print(f"\nnoise: {noise.shape}, time: {time.shape}, x_t: {x_t.shape}, u_t: {u_t.shape}")

    # (2) Embed Prefix (图像 + 语言)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation_p)
    act_mem_details['prefix_tokens'] = prefix_tokens.nbytes
    act_mem_details['prefix_mask'] = prefix_mask.nbytes
    act_mem_details['prefix_ar_mask'] = prefix_ar_mask.nbytes
    total_act_bytes += prefix_tokens.nbytes + prefix_mask.nbytes + prefix_ar_mask.nbytes
    print(f"\nprefix_tokens: {prefix_tokens.shape}, prefix_mask: {prefix_mask.shape}, prefix_ar_mask: {prefix_ar_mask.shape}")

    # (3) Embed Suffix (Action + timestep for adaRMS)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation_p, x_t, time)
    act_mem_details['suffix_tokens'] = suffix_tokens.nbytes
    act_mem_details['suffix_mask'] = suffix_mask.nbytes
    act_mem_details['suffix_ar_mask'] = suffix_ar_mask.nbytes
    # adarms_cond 如果是 None 则不计，否则记录（这里 pi05 有值）
    if adarms_cond is not None:
        act_mem_details['adarms_cond'] = adarms_cond.nbytes
        total_act_bytes += adarms_cond.nbytes
    total_act_bytes += suffix_tokens.nbytes + suffix_mask.nbytes + suffix_ar_mask.nbytes
    print(f"suffix_tokens: {suffix_tokens.shape}, suffix_mask: {suffix_mask.shape}, suffix_ar_mask: {suffix_ar_mask.shape}")
    print(f"adarms_cond: {None if adarms_cond is None else adarms_cond.shape}")

    # (4) 拼接并生成 Attention Mask
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    act_mem_details['input_mask'] = input_mask.nbytes
    act_mem_details['ar_mask'] = ar_mask.nbytes
    total_act_bytes += input_mask.nbytes + ar_mask.nbytes
    print(f"\nconcatenated input_mask: {input_mask.shape}, ar_mask: {ar_mask.shape}")

    from openpi.models.pi0 import make_attn_mask

    attn_mask = make_attn_mask(input_mask, ar_mask)  # 形状通常为 [B, 1, L, L]
    positions = jnp.cumsum(input_mask, axis=1) - 1
    act_mem_details['attn_mask'] = attn_mask.nbytes
    act_mem_details['positions'] = positions.nbytes
    total_act_bytes += attn_mask.nbytes + positions.nbytes
    print(f"attn_mask: {attn_mask.shape}, positions: {positions.shape}")

    # (5) Gemma LLM 双塔前向 (PaliGemma + Action Expert)
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    if prefix_out is not None:
        act_mem_details['prefix_out'] = prefix_out.nbytes
        total_act_bytes += prefix_out.nbytes
    act_mem_details['suffix_out'] = suffix_out.nbytes
    total_act_bytes += suffix_out.nbytes
    print(f"\nprefix_out: {None if prefix_out is None else prefix_out.shape}")
    print(f"suffix_out: {suffix_out.shape}")

    # (6) Action 输出投影
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    act_mem_details['v_t'] = v_t.nbytes
    total_act_bytes += v_t.nbytes
    print(f"v_t (predicted velocity): {v_t.shape}")

    # (7) Loss 计算
    loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
    act_mem_details['loss'] = loss.nbytes
    total_act_bytes += loss.nbytes
    print(f"\nper-step loss shape: {loss.shape}")
    print(f"mean loss: {jnp.mean(loss)}")

    # 打印详细的激活值显存占用
    print("\n=== Intermediate Activations Memory Breakdown (Forward Pass) ===")
    for name, bytes_ in act_mem_details.items():
        print(f"  {name}: {_human_readable_size(bytes_)} ({bytes_:,} bytes)")
    print(f"  TOTAL ACTIVATIONS (accumulated / upper bound): {_human_readable_size(total_act_bytes)} ({total_act_bytes:,} bytes)")

    # (8) 官方接口对比
    loss_ref = model.compute_loss(rng, observation, actions, train=False)
    print(f"\ncompute_loss() 官方接口结果 shape: {loss_ref.shape}, mean: {jnp.mean(loss_ref)}")

    # (9) 采样推理（不重复计入激活，仅展示）
    sampled_actions = model.sample_actions(jax.random.key(2), observation, num_steps=10)
    print(f"\nsample_actions() 输出 shape: {sampled_actions.shape}@{sampled_actions.dtype}")

    # --- 10. 全量 GPU 显存构成估算（训练场景） ---
    print("\n=== Full GPU Memory Breakdown Estimate (Training with Adam) ===")
    # 假设：所有参数、梯度和优化器状态使用与模型相同的 dtype（通常为 float32 或 bfloat16）
    # 这里依据 total_bytes 推断。如果使用混合精度，实际情况可能不同。
    grad_bytes = total_bytes  # 梯度通常与参数大小相同
    optim_bytes = total_bytes * 2  # Adam 优化器需要维护一阶矩(m)和二阶矩(v)，共 2 倍参数量

    total_est = total_bytes + grad_bytes + optim_bytes + total_act_bytes

    print(f"  Parameters:          {_human_readable_size(total_bytes)} ({total_bytes:,} bytes)")
    print(f"  Gradients:           {_human_readable_size(grad_bytes)} (same as params)")
    print(f"  Optimizer States:    {_human_readable_size(optim_bytes)} (Adam: 2x params for m & v)")
    print(f"  Activations (est.):  {_human_readable_size(total_act_bytes)} ({total_act_bytes:,} bytes) [Upper Bound]")
    print(f"  --------------------------------------------------")
    print(f"  TOTAL ESTIMATED:     {_human_readable_size(total_est)} ({total_est:,} bytes)")


if __name__ == "__main__":
    config, model, total_params, total_bytes = create_pi05()
    forward_pass(config, model, batch_size=2, total_params=total_params, total_bytes=total_bytes)
