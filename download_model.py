# SDK模型下载
from modelscope import snapshot_download
# snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', local_dir='pretrained_models/DeepSeek-R1-Distill-Qwen-7B')
# snapshot_download(model_id="sentence-transformers/all-MiniLM-L6-v2", local_dir='pretrained_models/all-MiniLM-L6-v2')
snapshot_download('lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF', local_dir='pretrained_models/DeepSeek-R1-Distill-Qwen-7B-GGUF')