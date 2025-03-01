# SDK模型下载
from modelscope import snapshot_download
# snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', local_dir='pretrained_models/DeepSeek-R1-Distill-Qwen-7B')
snapshot_download(model_id="AI-ModelScope/bge-large-zh", local_dir='pretrained_models/bge-large-zh')
# snapshot_download('lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF', local_dir='pretrained_models/DeepSeek-R1-Distill-Qwen-14B-GGUF')