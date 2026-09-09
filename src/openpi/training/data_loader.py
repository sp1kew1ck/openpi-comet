from collections.abc import Iterator, Sequence
import contextlib
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class StateActionDataset(Dataset[dict]):
    """Read only the columns required for normalization statistics."""

    def __init__(self, dataset, columns: Sequence[str]):
        self._dataset = dataset
        # Absolute frame indices are required to compute delta windows correctly.
        requested = [*columns, "index"] if "index" in dataset.features else list(columns)
        self._columns = [key for key in requested if key in dataset.features]
        missing = set(columns) - set(self._columns)
        if missing:
            raise KeyError(f"Normalization columns are missing from the dataset: {sorted(missing)}")
        self._dataset.hf_dataset = self._dataset.hf_dataset.select_columns(self._columns)

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = index.__index__()
        item = self._dataset.hf_dataset[index]
        ep_idx = item["episode_index"].item()

        if self._dataset.delta_indices is not None:
            # Delta windows live in the absolute frame-index space (see
            # LeRobotDataset.__getitem__), not in the loader's positional space. Using
            # the positional index here silently returns degenerate windows whenever the
            # loaded episodes do not start at the beginning of the dataset.
            abs_idx = item["index"].item() if "index" in item else index
            query_indices, padding = self._dataset._get_query_indices(abs_idx, ep_idx)
            item.update(padding)
            item.update(self._dataset._query_hf_dataset(query_indices))

        return item

    def __len__(self) -> int:
        return len(self._dataset)


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def tasks_to_mapping(tasks) -> dict[int, str]:
    """Normalize LeRobot task metadata into a {task_index: task name} mapping.

    Older LeRobot versions expose tasks as a dict, while newer ones (>= 0.4) store
    ``meta/tasks.parquet`` as a pandas DataFrame whose task names are either a "task"
    column or the index, alongside the "task_index" column.
    """
    if isinstance(tasks, dict):
        return {int(index): str(task) for index, task in tasks.items()}
    import pandas as pd

    if isinstance(tasks, pd.DataFrame):
        if tasks.index.name == "task":
            tasks = tasks.reset_index()
        if "task" in tasks.columns:
            return {int(row["task_index"]): str(row["task"]) for _, row in tasks.iterrows()}
        # Task names live in the (unnamed) DataFrame index.
        return {int(row["task_index"]): str(task) for task, row in tasks.iterrows()}
    raise TypeError(f"Unsupported tasks metadata type: {type(tasks)}")


class _ValueIndexedEpisodes:
    """Episodes metadata addressed by ``episode_index`` *value*, not by row position.

    LeRobot >= 0.4 indexes ``meta.episodes[episode_index]`` positionally, which only
    holds when episodes are numbered 0..N-1 in row order. Datasets built as subsets of a
    larger dataset (e.g. scripts/build_b1k_subset.py) keep the source numbering, so
    their rows are not addressable by their own ``episode_index``. This wrapper leaves
    the original rows untouched and redirects integer lookups (and the ``len()`` bound
    used by LeRobot's guards) to the value space.
    """

    def __init__(self, table):
        self._table = table
        self._position = {int(index): i for i, index in enumerate(table["episode_index"])}

    def __getitem__(self, index):
        return self._table[self._position[int(index)]]

    def __len__(self) -> int:
        # LeRobot validates episode indices with `ep_index >= len(episodes)`.
        return max(self._position) + 1 if self._position else 0

    def __getattr__(self, name):
        # Delegate only regular public attributes to the underlying table. Never resolve
        # private/dunder names through it: pickle probes for e.g. `__setstate__` while
        # restoring a bare instance (created via __new__, before `_table` exists), and an
        # unguarded delegation used to recurse infinitely through `self._table`.
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
        table = self.__dict__.get("_table")
        if table is None:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
        return getattr(table, name)


@contextlib.contextmanager
def _local_dataset_guards(root):
    """LeRobot may silently fall back to hub downloads whenever a local check fails.

    For local (read-only) dataset roots that is never desired, so fail fast instead.
    While active, metadata episodes tables are also wrapped to be addressable by value,
    which is a no-op for canonical (0..N-1) datasets.
    """
    if root is None:
        yield
        return
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    original_load_episodes = lerobot_dataset.load_episodes

    def load_episodes(local_dir):
        episodes = original_load_episodes(local_dir)
        try:
            indices = [int(i) for i in episodes["episode_index"]]
        except (AttributeError, KeyError, TypeError):
            return episodes  # Not a columnar episodes table: nothing to align.
        if indices != list(range(len(episodes))):
            episodes = _ValueIndexedEpisodes(episodes)
        return episodes

    lerobot_dataset.load_episodes = load_episodes
    try:
        yield
    finally:
        lerobot_dataset.load_episodes = original_load_episodes
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline


def _resolve_episodes(dataset_meta, data_config: _config.DataConfig) -> list[int]:
    """Episodes to load, expressed in the dataset's own numbering.

    ``episodes_index`` selects explicitly; otherwise all episodes present in the
    metadata are returned. Passing ``None`` to LeRobot would request ``0..total-1``,
    which breaks datasets whose numbering is not row aligned (B1K subsets built by
    scripts/build_b1k_subset.py preserve the source episode numbering).

    When ``tasks`` (task names) is set, only episodes of these tasks are kept.
    """
    if data_config.episodes_index is None:
        episodes = sorted(int(i) for i in dataset_meta.episodes["episode_index"])
    else:
        episodes = sorted(int(i) for i in data_config.episodes_index)
    if not data_config.tasks:
        return episodes
    name_to_index = {name: index for index, name in tasks_to_mapping(dataset_meta.tasks).items()}
    unknown = sorted(set(data_config.tasks) - set(name_to_index))
    if unknown:
        raise ValueError(f"Tasks not found in dataset: {unknown}")
    table = dataset_meta.episodes
    if "task_index" not in (getattr(table, "column_names", None) or ()):
        raise ValueError("Cannot filter by tasks: dataset episodes have no 'task_index' column")
    task_indices = {name_to_index[name] for name in data_config.tasks}
    task_episodes = {
        int(i) for i, t in zip(table["episode_index"], table["task_index"]) if int(t) in task_indices
    }
    episodes = sorted(set(episodes) & task_episodes)
    if not episodes:
        raise ValueError(f"No episodes found for tasks {data_config.tasks}")
    logging.info("Filtered episodes to %d for tasks %s", len(episodes), sorted(task_indices))
    return episodes


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    load_visual_data: bool = True,
    stats_columns: Sequence[str] | None = None,
) -> Dataset:
    """Create a LeRobot dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Use local root when behavior_dataset_root is set (B1K datasets stored locally).
    root = data_config.behavior_dataset_root or None
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    episodes = _resolve_episodes(dataset_meta, data_config)

    with _local_dataset_guards(root):
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
        )

    if not load_visual_data:
        if stats_columns is None:
            raise ValueError("stats_columns must be set when load_visual_data is False")
        dataset = StateActionDataset(dataset, stats_columns)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(tasks_to_mapping(dataset_meta.tasks))]
        )

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> IterableDataset:
    """Create an RLDS dataset for training (currently only supports DROID).

    Requires the optional droid_rlds_dataset module and its dependencies (tensorflow, dlimp, etc.).
    """
    from openpi.training.droid_rlds_dataset import DroidActionSpace as _DroidActionSpace
    from openpi.training.droid_rlds_dataset import DroidRldsDataset, RLDSDataset

    # Map the config enum to droid_rlds_dataset's enum by name so the two definitions stay independent.
    action_space = None
    if data_config.action_space is not None:
        action_space = _DroidActionSpace[data_config.action_space.name]

    # Build a single dataset entry from the DataConfig fields.
    datasets = [
        RLDSDataset(
            name=data_config.repo_id or "droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path=data_config.filter_dict_path,
        )
    ]

    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=action_space,
        datasets=datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def _create_behavior_dataset(
    config: _config.TrainConfig,
) -> tuple[Dataset, _config.DataConfig]:
    """Build the (possibly multi-config) raw torch dataset for a training config."""
    if isinstance(config.data, list):
        if config.sample_weights and any(w != config.sample_weights[0] for w in config.sample_weights[1:]):
            raise NotImplementedError(
                "Non-uniform sample_weights require the legacy omnigibson MultiBehaviorLeRobotDataset."
            )
        data_configs = [data.create(config.assets_dirs, config.model) for data in config.data]
        datasets = [create_torch_dataset(dc, config.model.action_horizon, config.model) for dc in data_configs]
        return torch.utils.data.ConcatDataset(datasets), data_configs[0]
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    return dataset, data_config


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    seed_shift: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training (single- or multi-config)."""
    dataset, data_config = _create_behavior_dataset(config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed + seed_shift,
    )
    return DataLoaderImpl(data_config, data_loader)


def create_torch_behavior_data_loader(
    config: _config.TrainConfig,
    action_horizon: int,
    batch_size: int,
    *,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a PyTorch-framework data loader for training."""
    if config.model.action_horizon != action_horizon:
        raise ValueError(f"action_horizon {action_horizon} != model action_horizon {config.model.action_horizon}")
    dataset, data_config = _create_behavior_dataset(config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = batch_size

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=num_workers,
        seed=seed,
        framework="pytorch",
    )
    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        # Use a process-index-dependent seed so each JAX process draws different samples.
        process_seed = seed + jax.process_index()
        generator = torch.Generator()
        generator.manual_seed(process_seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration as e:
                    logging.info(f"Stop Iteration ... {e}")
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
