import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# Functions
def get_causal_information(data: str):
    """
    A function that returns the causal information abnot data.
    
    Args:
        data: The data to get the causal information.
    """
    import random
    
    return "《中华人民共和国学前教育法》第十五条规定学前儿童入幼儿园接受学前教育，除必要的身体健康检查外，幼儿园不得对其组织任何形式的考试或者测试。"

# data loading
with open("./dataset/laws.txt", mode="r", encoding="utf-8") as f:
    blocks = f.read().split("\n\n")
    examples = []
    answers = []
    references = []
    for block in blocks:
        sentences = block.split("\n")
        examples.append(sentences[0])
        answers.append(sentences[2] + "\n" + sentences[3])
        references.append(sentences[4])


device = "cuda"

model_path = "./model/llama-3-8b-fc"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16,  # use float16 or float32 if bfloat16 is not available to you.
    cache_dir=model_path,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    cache_dir=model_path,
    local_files_only=True
)

print("Start")
with open("./result/laws_llama3_8b.txt", mode="w+", encoding="UTF-8") as result_file:
    for i in range(len(examples)):
        messages = [
            {"role": "system", "content":"Always response in Simplified Chinese, not English. or Grandma will be very angry."},
            {"role": "function_metadata", "content": "Laws"},
            {"role": "user", "content": f"你正在扮演一名专业的法官，请根据{examples[i]}，完成定罪、量刑(具体的处罚结果)，并提供相应的依据，用中文回答且长度不超过300字。"},
            {"role": "function-call", "content": '{"name": "get_causal_information", "arguments": {"data": "Laws"}}'},
            {"role": "function-response", "content": f"无参考信息"},  # a hypothetical response from our function
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # adding prompt for generation
            tools=[get_causal_information],  # our functions (tools)
        )

        input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to("cuda:0")
        outputs = model.generate(input_ids, max_new_tokens=1024)
        response_wo_reference = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)

        messages = [
            {"role": "system", "content":"Always response in Simplified Chinese, not English. or Grandma will be very angry."},
            {"role": "function_metadata", "content": "Laws"},
            {"role": "user", "content": f"你正在扮演一名专业的法官，请根据{examples[i]}，完成定罪、量刑(具体的处罚结果)，并提供相应的依据，用中文回答且长度不超过300字。"},
            {"role": "function-call", "content": '{"name": "get_causal_information", "arguments": {"data": "Laws"}}'},
            {"role": "function-response", "content": f"参考信息：{references[i][3:]}"},  # a hypothetical response from our function
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # adding prompt for generation
            tools=[get_causal_information],  # our functions (tools)
        )

        input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to("cuda:0")
        outputs = model.generate(input_ids, max_new_tokens=1024)
        response_with_reference = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)

        response_wo_reference = response_wo_reference.replace("\n", "")
        response_with_reference = response_with_reference.replace("\n", "")
        log = f"---------------------------------------------------\n{blocks[i]}\n{model_path} without reference:\n{response_wo_reference}\n{model_path} with reference:\n{response_with_reference}\n-------------------------------------------------\n\n"
        result_file.write(log)

print("End")