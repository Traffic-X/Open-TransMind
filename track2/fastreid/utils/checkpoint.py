# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import logging
import os
from collections import defaultdict
from typing import Any
from typing import Optional, List, Dict, NamedTuple, Tuple, Iterable
from subprocess import Popen

import numpy as np
import paddle
from termcolor import colored

from fastreid.utils.file_io import PathManager


class _IncompatibleKeys(
    NamedTuple(
        # pyre-fixme[10]: Name `IncompatibleKeys` is used but not defined.
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            ("incorrect_shapes", List[Tuple]),
        ],
    )
):
    pass


class Checkpointer(object):
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.
    """

    def __init__(
            self,
            model=None,
            save_dir="",
            save_to_disk=True,
            **kwargs,
    ):
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer.
            kwargs (object): any checkpointable objects, i.e., objects
                that have the `state_dict()` and `load_state_dict()` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        if isinstance(model, paddle.DataParallel):
            model = model._layers
        self.model = model
        self.checkpointables = copy.copy(kwargs)
        self.logger = logging.getLogger(__name__)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

        self.path_manager = PathManager

    def save(self, name: str, **kwargs):
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pdmodel".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        paddle.save(data, save_file)
        self.tag_last_checkpoint(basename)
        # p = Popen(['sh', 'scripts/upload_afs.sh', save_file])
        # p.wait()

    def load(self, path: str, checkpointables=None):
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Training model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        if "model" in checkpoint:
            checkpoint_state_dict = checkpoint.pop("model")
        else:
            checkpoint_state_dict = checkpoint
        self.model.set_state_dict(checkpoint_state_dict)
        missing_keys = []
        unexpected_keys = []
        for k, v in self.model.state_dict().items():
            if k not in checkpoint_state_dict.keys():
                missing_keys.append(k)
            else:
                if v.shape != checkpoint_state_dict[k].shape:
                    self.logger.warn('{} has {}, but {} has {}'.format(k, v.shape, k, checkpoint_state_dict[k].shape))
        for k, v in checkpoint_state_dict.items():
            if k not in self.model.state_dict().keys():
                unexpected_keys.append(k)
            else:
                if v.shape != self.model.state_dict()[k].shape:
                    self.logger.warn('{} has {}, but {} has {}'.format(k, v.shape, k, self.model.state_dict()[k].shape))
        self.logger.warn(f"missing keys: {missing_keys}")
        self.logger.warn(f"unexpected keys: {unexpected_keys}")
        # incompatible = self._load_model(checkpoint)
        # if (
        #         incompatible is not None
        # ):  # handle some existing subclasses that returns None
        #     self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:  # pyre-ignore
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.set_state_dict(checkpoint.pop(key))  # pyre-ignore

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return PathManager.exists(save_file)

    def get_checkpoint_file(self):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with PathManager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)

    def get_all_checkpoint_files(self):
        """
        Returns:
            list: All available checkpoint files (.pdmodel files) in target
                directory.
        """
        all_model_checkpoints = [
            os.path.join(self.save_dir, file)
            for file in PathManager.ls(self.save_dir)
            if PathManager.isfile(os.path.join(self.save_dir, file))
               and file.endswith(".pdmodel")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool = True):
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists.
        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])

    def tag_last_checkpoint(self, last_filename_basename: str):
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with PathManager.open(save_file, "w") as f:
            f.write(last_filename_basename)

    def _load_file(self, f):
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.

        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to Tensor or numpy arrays.
        """
        return paddle.load(f)

    def _load_model(self, checkpoint):
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "_layers.")

        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)

        # incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        incompatible = self.model.set_state_dict(checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

    def _log_incompatible_keys(self, incompatible) -> None:
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        for k, shape_checkpoint, shape_model in incompatible.incorrect_shapes:
            self.logger.warning(
                "Skip loading parameter '{}' to the model due to incompatible "
                "shapes: {} in the checkpoint but {} in the "
                "model! You might want to double check if this is expected.".format(
                    k, shape_checkpoint, shape_model
                )
            )
        if incompatible.missing_keys:
            missing_keys = _filter_reused_missing_keys(
                self.model, incompatible.missing_keys
            )
            if missing_keys:
                self.logger.info(get_missing_parameters_message(missing_keys))
        if incompatible.unexpected_keys:
            self.logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    def _convert_ndarray_to_tensor(self, state_dict):
        """
        In-place convert all numpy arrays in the state_dict to tensor.

        Args:
            state_dict (dict): a state-dict to be loaded to the model.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(
                    v, paddle.Tensor
            ):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(
                        k, type(v)
                    )
                )
            if not isinstance(v, paddle.Tensor):
                state_dict[k] = paddle.to_tensor(v)


def _filter_reused_missing_keys(model, keys):
    """
    Filter "missing keys" to not include keys that have been loaded with another name.
    """
    keyset = set(keys)
    param_to_names = defaultdict(set)  # param -> names that points to it
    for module_prefix, module in _named_modules_with_dup(model):
        for name, param in list(module.named_parameters(recurse=False)) + list(
                module.named_buffers(recurse=False)  # pyre-ignore
        ):
            full_name = (module_prefix + "." if module_prefix else "") + name
            param_to_names[param].add(full_name)
    for names in param_to_names.values():
        # if one name appears missing but its alias exists, then this
        # name is not considered missing
        if any(n in keyset for n in names) and not all(n in keyset for n in names):
            [keyset.remove(n) for n in names if n in keyset]
    return list(keyset)


def get_missing_parameters_message(keys):
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.

    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys):
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.

    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _strip_prefix_if_present(state_dict, prefix):
    """
    Strip the prefix in metadata, if any.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys):
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.

    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group):
    """
    Format a group of parameter name suffixes into a loggable string.

    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


def _named_modules_with_dup(
        model, prefix=""
):
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():  # pyre-ignore
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)
