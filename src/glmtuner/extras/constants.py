IGNORE_INDEX = -100

VALUE_HEAD_FILE_NAME = "value_head.bin"

FINETUNING_ARGS_NAME = "finetuning_args.json"

LAYERNORM_NAMES = ["layernorm"]

METHODS = ["full", "freeze", "p_tuning", "lora"]

SUPPORTED_MODELS = {
    "ChatGLM-6B": "THUDM/chatglm-6b",
    # "ChatGLM2-6B": "THUDM/chatglm2-6b"
    "ChatGLM2-6B": "/root/autodl-tmp/chatglm2-6b"
}
