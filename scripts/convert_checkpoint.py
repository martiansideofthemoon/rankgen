import os
import re
import subprocess
import time
import functools
import sys
import pdb
from typing import Any, Dict, Iterable, MutableMapping, Mapping, Optional, Sequence, Tuple, List

import asyncio
import pickle
import transformers
from absl import logging
from flax import optim
from flax import serialization
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from t5x import checkpoint_importer
from t5x import multihost_utils
from t5x import state_utils
from t5x import train_state as train_state_lib
import tensorflow as tf
from tensorflow.io import gfile
import tensorstore as ts
import numpy as np
import typing_extensions
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper

from t5x.checkpoints import _maybe_update_ts_from_file_to_gcs, _maybe_update_ts_from_gcs_to_file, \
    RestoreStateTransformationFn

PyTreeDef = type(jax.tree_structure(None))
LazyArray = checkpoint_importer.LazyArray
LazyAwaitableArray = checkpoint_importer.LazyAwaitableArray
LazyThreadPoolArray = checkpoint_importer.LazyThreadPoolArray

VERSION = 3
_DESIRED_CHUNK_SIZE_BYTES = 64 * 1024 * 1024


class _ParameterInfo:
    name: str
    ts_spec: Optional[ts.Spec]

    def __init__(self, name, ts_spec):
        self.name = name
        self.ts_spec = ts_spec


def _get_optimizer_state_dict(
        ckpt_contents: PyTreeDef, optimizer_state: Mapping[str, Any],
        state_transformation_fns: Sequence[RestoreStateTransformationFn]):
    version = ckpt_contents.get('version', 0)
    if version == 0:
        ckpt_optimizer_state = ckpt_contents
    else:
        ckpt_optimizer_state = ckpt_contents['optimizer']

    if version >= 2:
        for fn in state_transformation_fns:
            ckpt_optimizer_state = fn(ckpt_optimizer_state, optimizer_state)
        return ckpt_optimizer_state
    else:
        raise ValueError('Checkpoint versions earlier than 2 are not supported. '
                         f'Got version: {version}')


def _cast(target: PyTreeDef, dtype: jnp.dtype):
    def maybe_cast(x):
        if isinstance(x, (int, str)):
            # Ignore common non-array types that shouldn't be cast.
            return x
        elif x.dtype == dtype:
            return x
        elif isinstance(x, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(x.shape, dtype)
        else:
            return x.astype(dtype)

    return jax.tree_map(maybe_cast, target)


def _get_state_dict_for_save(state_dict: Dict[str, Any],
                             lazy_load: bool = True) -> Mapping[str, Any]:
    def _lazy_load_device_array(arr):
        if isinstance(arr, jax.xla.DeviceArray):
            return LazyThreadPoolArray(arr.shape, arr.dtype, lambda: np.array(arr))
        return arr

    if lazy_load:
        state_dict = jax.tree_map(_lazy_load_device_array, state_dict)
    state_dict['target'] = _cast(state_dict['target'], np.float32)
    return state_dict


def _get_parameter_infos(ckpt_state_dict):
    def _get_param_info(name: str, arr: Any):
        # print("PRINTING")
        # print(name)
        # print(arr)
        # print(type(arr))
        # if type(arr) == 'numpy.ndarray':
        return _ParameterInfo(name=name, ts_spec=None)

    param_names = traverse_util.unflatten_dict({
        k: '/'.join(k) for k in traverse_util.flatten_dict(
            ckpt_state_dict, keep_empty_nodes=True)
    })

    # print(param_names)
    # print(_get_state_dict_for_save(ckpt_state_dict))

    return jax.tree_map(
        _get_param_info, param_names,
        _get_state_dict_for_save(ckpt_state_dict))


async def _read_ts(param_info: _ParameterInfo, maybe_tspec: Any,
                   ckpt_path: str):
    # If saved as a numpy array, but a partitioned read is requested, return a
    # slice of the array for that host. Otherwise, return the whole thing.
    if isinstance(maybe_tspec, np.ndarray) and param_info:
        return maybe_tspec
    # If we have anything else that isn't a tensorstore spec just return it.
    elif not isinstance(maybe_tspec, ts.Spec):
        return maybe_tspec

    tmp_ts_spec_dict = maybe_tspec.to_json()
    # Remove non-required params so that we can open Tensorstore
    # that was created with a different set of params.
    del tmp_ts_spec_dict['metadata']['chunks']
    del tmp_ts_spec_dict['metadata']['compressor']

    # Convert the relative path in the spec to a path based on the checkpoint
    # location. Path and gcs bucket (if applicable) information is updated
    # in-place.
    _update_ts_path_from_relative_to_absolute(
        os.path.dirname(ckpt_path), tmp_ts_spec_dict)

    # if param_info.shape is not None:
    #     ts_spec_arr_shape = tuple(tmp_ts_spec_dict['metadata']['shape'])
    #     # Check that the shapes of the array on disk match the expected shape based
    #     # on the optimizer that is being restored.
    #     if ts_spec_arr_shape != param_info.shape:
    #         raise ValueError(f'Shape of `{param_info.name}` in checkpoint '
    #                          f'{ts_spec_arr_shape} does not match expected '
    #                          f'{param_info.shape}.')
    # Read the array.
    t = await ts.open(tmp_ts_spec_dict, open=True)
    if param_info.local_chunk_info is not None:
        # Just read the subsection we care about.
        t = t[param_info.local_chunk_info.slice]
    arr = await t.read()
    # Assume we had to cast bfloat16 to uint16 to store with zarr.
    # TODO(ndl): remove this bitcast, as well as related bitcasts in PW code,
    # once we're ready to deprecate T5X checkpoints with "legacy" bfloat16
    # support.
    if arr.dtype == np.uint16:
        arr = arr.view(jnp.bfloat16)
    return arr


def _create_lazy_awaitable_array(param_info: _ParameterInfo, maybe_ts_spec: Any,
                                 ckpt_path: str) -> LazyAwaitableArray:
    get_fn = functools.partial(
        _read_ts, param_info, maybe_ts_spec, ckpt_path=ckpt_path)
    if isinstance(maybe_ts_spec, ts.Spec) or isinstance(maybe_ts_spec, np.ndarray):
        return LazyAwaitableArray.from_tensor_store_spec_or_array(
            maybe_ts_spec, get_fn)


def _read_state_from_tensorstore(
        ckpt_path: str,
        parameter_infos: _ParameterInfo,
        written_state_dict: Mapping[str, Any],
        restore_parameter_infos: Optional[Mapping[str, Any]] = None,
        lazy_parameters: bool = False
) -> Mapping[str, Any]:
    if restore_parameter_infos is None:
        restore_parameter_infos = parameter_infos

    # Replace TensorStore Specs with the lazy array values.
    state_dict = jax.tree_multimap(
        functools.partial(_create_lazy_awaitable_array, ckpt_path=ckpt_path),
        restore_parameter_infos, written_state_dict)

    if not lazy_parameters:
        future_state_dict = jax.tree_map(lambda x: x.get_async(), state_dict)
        state_dict = _run_future_tree(future_state_dict)

    state_dict['target'] = _cast(state_dict['target'], np.float32)

    return state_dict


def _run_future_tree(future_tree):
    """Block until all futures are resolved on this host."""
    future_leaves, treedef = jax.tree_flatten(future_tree)

    # TODO(adarob): Use asyncio.run in py3.7+.
    loop = asyncio.get_event_loop()
    leaves = loop.run_until_complete(asyncio.gather(*future_leaves))
    return jax.tree_unflatten(treedef, leaves)


def restore(
        path: Optional[str] = None,
        fallback_state: Optional[Mapping[str, Any]] = None,
        lazy_parameters: bool = False) -> train_state_lib.TrainState:
    ckpt_path = path

    if gfile.isdir(ckpt_path):
        ckpt_dir = ckpt_path
        ckpt_path = os.path.join(ckpt_path, 'checkpoint')
    else:
        ckpt_dir = os.path.dirname(ckpt_path)

    if not gfile.exists(ckpt_path) or gfile.isdir(ckpt_path):
        raise ValueError(f'Path is not a valid T5X checkpoint: {ckpt_path}')

    logging.info('Restoring from checkpoint: %s', ckpt_path)

    with gfile.GFile(ckpt_path, 'rb') as fp:
        raw_contents = fp.read()
        if raw_contents.startswith(b'model_checkpoint_path'):
            raise ValueError(
                'Attempting to restore a TensorFlow checkpoint as a native T5X '
                'checkpoint. Use `restore_from_tf_checkpoint` instead. Path: ' +
                ckpt_path)

        ckpt_contents = serialization.msgpack_restore(raw_contents)

    if ckpt_dir.startswith('gs://'):
        ckpt_contents = _maybe_update_ts_from_file_to_gcs(ckpt_contents)
    else:
        ckpt_contents = _maybe_update_ts_from_gcs_to_file(ckpt_contents)

    ckpt_state_dict = _get_optimizer_state_dict(ckpt_contents, [], [])

    # print(ckpt_state_dict)

    dummy_spec = ts.Spec({'driver': 'zarr', 'kvstore': {'driver': 'memory'}})

    parameter_infos = _get_parameter_infos(ckpt_state_dict)
    # print(parameter_infos)
    dummy_written_state_dict = jax.tree_map(
        lambda x: x.ts_spec or dummy_spec,
        parameter_infos,
    )

    if fallback_state is None:
        restore_parameter_infos = parameter_infos
    else:
        dummy_written_state_dict = state_utils.intersect_state(
            dummy_written_state_dict, ckpt_state_dict)
        restore_parameter_infos = state_utils.intersect_state(
            _parameter_infos, ckpt_state_dict)

    restore_parameter_infos_flat = state_utils.flatten_state_dict(
        restore_parameter_infos)
    for key in restore_parameter_infos_flat.keys():
        logging.info('Restoring key from ckpt: %s', key)

    written_state_dict = serialization.from_state_dict(dummy_written_state_dict,
                                                       ckpt_state_dict)
    state_dict = _read_state_from_tensorstore(
        ckpt_path,
        parameter_infos,
        written_state_dict,
        restore_parameter_infos=restore_parameter_infos,
        lazy_parameters=lazy_parameters)

    if fallback_state is not None:
        state_dict = state_utils.merge_state(state_dict, fallback_state)

    for key in state_utils.flatten_state_dict(state_dict).keys():
        if key not in restore_parameter_infos_flat:
            logging.info('Not restoring key from ckpt: %s', key)

    return ckpt_state_dict


def read_array(data):
    path = data['kvstore']['path']
    if path.startswith('t5x/pre_suf_retriever/checkpoint_1100000/') == False:
        data['kvstore']['path'] = 't5x/pre_suf_retriever/checkpoint_1100000/' + path
    dataset = ts.open(data).result()
    return np.array(dataset)


ckpt = restore(path="t5x/pre_suf_retriever/checkpoint_1100000")
state_dict = {}

state_dict['encoder.final_layer_norm.weight'] = ckpt['target']['encoder']['encoder_norm']['scale']
state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] = ckpt['target']['encoder']['relpos_bias']['rel_embedding']
state_dict['encoder.embed_tokens.weight'] = read_array(ckpt['target']['token_embedder']['embedding'])

for key, value in ckpt['target']['encoder'].items():
    if key.startswith('layers_'):
        n = key[7:]
        state_dict[f'encoder.block.{n}.layer.0.layer_norm.weight'] = value['pre_attention_layer_norm']['scale']
        state_dict[f'encoder.block.{n}.layer.0.SelfAttention.k.weight'] = read_array(
            value['attention']['key']['kernel'])
        state_dict[f'encoder.block.{n}.layer.0.SelfAttention.q.weight'] = read_array(
            value['attention']['query']['kernel'])
        state_dict[f'encoder.block.{n}.layer.0.SelfAttention.v.weight'] = read_array(
            value['attention']['value']['kernel'])
        state_dict[f'encoder.block.{n}.layer.0.SelfAttention.o.weight'] = read_array(
            value['attention']['out']['kernel'])
        state_dict[f'encoder.block.{n}.layer.1.layer_norm.weight'] = value['pre_mlp_layer_norm']['scale']
        state_dict[f'encoder.block.{n}.layer.1.DenseReluDense.wi_0.weight'] = read_array(
            value['mlp']['wi_0']['kernel'])
        state_dict[f'encoder.block.{n}.layer.1.DenseReluDense.wi_1.weight'] = read_array(
            value['mlp']['wi_1']['kernel'])
        state_dict[f'encoder.block.{n}.layer.1.DenseReluDense.wo.weight'] = read_array(
            value['mlp']['wo']['kernel'])

with open('state_dict.pickle', 'wb') as handle:
    pickle.dump(state_dict, handle)

projection = ckpt['target']['encoder']['encoder_projection_layer']['kernel']
with open('projection.pickle', 'wb') as handle:
    pickle.dump(projection, handle)