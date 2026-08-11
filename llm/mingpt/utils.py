
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # 如果工作目录不存在则创建
    os.makedirs(work_dir, exist_ok=True)
    # 记录参数（如果有）
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # 记录配置本身
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO：转成像 yacs 里那样的 dict 子类？
    # TODO：实现冻结，防止搬起石头砸自己的脚
    # TODO：读写参数时增加存在性/覆盖检查？

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # 解包

            # 先把 val 转成 python 对象
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # 找到合适的对象来插入属性
            assert key[:2] == '--'
            key = key[2:] # 去掉 '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # 确保这个属性存在
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # 覆盖该属性
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
