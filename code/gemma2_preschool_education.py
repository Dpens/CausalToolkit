import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
with open("./dataset/laws_preschool_education.txt", mode="r", encoding="utf-8") as f:
    blocks = f.read().split("\n\n")
    answers = []
    references = []
    questions = []
    for block in blocks:
        sentences = block.split("\n")
        answers.append(sentences[2])
        references.append(sentences[3])
        questions.append(sentences[0])


device = "cuda"

# model_path = "./model/gemma-2-9b-it"
model_path = "./model/gemma-2-9b-it-fc"
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
with open("./result/laws_preschool_education_gemma2_9b.txt", mode="w+", encoding="UTF-8") as result_file:
    for i in range(len(questions)):
        # Function Calling
        history_messages = [
            {"role": "system", "content": "You are a helpful assistant with access to the following functions. Use them if required - "},
            {"role": "user", "content": f"你是一名中华人民共和国的儿童教育方面的法律专家，回答以下{questions[i]}并给出解释。回答长度不超过300字。"},
            {"role": "function-call", "content": '{"name": "get_causal_information", "arguments": {"data": "Laws"}}'},
            {"role": "function-response", "content": f"无参考信息"},  # a hypothetical response from our function
        ]
        inputs = tokenizer.apply_chat_template(
            history_messages,
            tokenize=False,
            add_generation_prompt=True,  # adding prompt for generation
            tools=[get_causal_information],  # our functions (tools)
        )

        terminator_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]
        prompt_ids =  tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt').to(model.device)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=512,
            eos_token_id=terminator_ids,
            bos_token_id=tokenizer.bos_token_id,
        )
        response_wo_reference = tokenizer.decode(generated_ids[0][prompt_ids.shape[-1]:], skip_special_tokens=False)  # `skip_special_tokens=False` for debug

        history_messages = [
            {"role": "system", "content": "You are a helpful assistant with access to the following functions. Use them if required - "},
            {"role": "user", "content": f"你是一名中华人民共和国的儿童教育方面的法律专家，回答以下{questions[i]}并给出解释。回答长度不超过300字。"},
            {"role": "function-call", "content": '{"name": "get_causal_information", "arguments": {"data": "Laws"}}'},
            {"role": "function-response", "content": f"{references[i]}"},  # a hypothetical response from our function
        ]
        inputs = tokenizer.apply_chat_template(
            history_messages,
            tokenize=False,
            add_generation_prompt=True,  # adding prompt for generation
            tools=[get_causal_information],  # our functions (tools)
        )

        terminator_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]
        prompt_ids =  tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt').to(model.device)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=512,
            eos_token_id=terminator_ids,
            bos_token_id=tokenizer.bos_token_id,
        )
        response_with_reference = tokenizer.decode(generated_ids[0][prompt_ids.shape[-1]:], skip_special_tokens=False)  # `skip_special_tokens=False` for debug

        response_wo_reference = response_wo_reference.replace("\n", "")
        response_with_reference = response_with_reference.replace("\n", "")
        log = f"---------------------------------------------------\n{blocks[i]}\n{model_path} without reference:\n{response_wo_reference}\n{model_path} with reference:\n{response_with_reference}\n-------------------------------------------------\n\n"
        result_file.write(log)
