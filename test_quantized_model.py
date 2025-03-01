from llama_cpp import Llama
import time

time_before = time.time()
llm = Llama(model_path="./pretrained_models/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
output = llm("Hello, World!", max_tokens=50)
print(output)
time_after = time.time()
total_time = time_after - time_before
print(total_time)
# llm = Llama.from_pretrained(
# 	repo_id="lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF",
# 	filename="DeepSeek-R1-Distill-Qwen-7B-Q3_K_L.gguf",
# )
#
# llm.create_chat_completion(
# 	messages = [
# 		{
# 			"role": "user",
# 			"content": "What is the capital of France?"
# 		}
# 	]
# )