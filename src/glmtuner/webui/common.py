import json
import os
from typing import Any, Dict, Optional

import gradio as gr
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
from transformers.trainer import WEIGHTS_NAME, WEIGHTS_INDEX_NAME

from glmtuner.extras.constants import SUPPORTED_MODELS
from glmtuner.extras.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = "cache"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user.config"
DATA_CONFIG = "dataset_info.json"


def get_save_dir(model_name: str) -> str:
    return os.path.join(DEFAULT_SAVE_DIR, os.path.split(model_name)[-1])


def get_config_path() -> os.PathLike:
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def load_config() -> Dict[str, Any]:
    try:
        with open(get_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"last_model": "", "path_dict": {}}


def save_config(model_name: str, model_path: str) -> None:
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["last_model"] = model_name
    user_config["path_dict"][model_name] = model_path
    with open(get_config_path(), "w", encoding="utf-8") as f:
        json.dump(user_config, f, indent=2, ensure_ascii=False)


def get_model_path(model_name: str) -> str:
    user_config = load_config()
    return user_config["path_dict"].get(model_name, SUPPORTED_MODELS.get(model_name, ""))


def list_checkpoint(model_name: str, finetuning_type: str) -> Dict[str, Any]:
    checkpoints = []
    save_dir = os.path.join(get_save_dir(model_name), finetuning_type)
    logger.info(
        "model_name is :{},finetuning_type is: {},save_dir is:{}, os.getcwd() is:{}".format(model_name, finetuning_type,
                                                                                            save_dir, os.getcwd()))
    if save_dir and os.path.isdir(save_dir):
        for checkpoint in os.listdir(save_dir):
            if (
                    os.path.isdir(os.path.join(save_dir, checkpoint))
                    and any([
                os.path.isfile(os.path.join(save_dir, checkpoint, name))
                for name in (WEIGHTS_NAME, WEIGHTS_INDEX_NAME, PEFT_WEIGHTS_NAME)
            ])
            ):
                checkpoints.append(checkpoint)
    return gr.update(value=[], choices=checkpoints)


def load_dataset_info(dataset_dir: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def list_dataset(dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    return gr.update(value=[], choices=list(dataset_info.keys()))
