uv add torch --index pytorch-cu128=https://download.pytorch.org/whl/cu128
uv add transformers accelerate optimum datasets
uv add "gptqmodel>=7.0.0" --no-build-isolation-package gptqmodel # 普通に入れると古すぎてエラー出たので下限を設定

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true <-- 書き足すこと