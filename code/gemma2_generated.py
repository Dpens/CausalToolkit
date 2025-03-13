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

for dataset in ["polynomial", "linear"]:
    # data loading
    with open(f"./dataset/{dataset}_generated_data.txt", mode="r", encoding="utf-8") as f:
        blocks = f.read().split("\n\n")
        answers = []
        references = []
        questions = []
        for block in blocks:
            sentences = block.split("\n")
            answers.append(sentences[2])
            references.append(sentences[1])
            questions.append(sentences[0])

    all_reference4one_question = ["无参考信息"]
    with open(f"./dataset/{dataset}_reference.txt", mode="r", encoding="utf-8") as f:
        all_reference4one_question += f.read().split("\n\n")

    with open(f"./result/{dataset}_generated_data_gemma2_9b.txt", mode="w+", encoding="UTF-8") as result_file:
        for i in range(len(questions)):
            tmp_reference = all_reference4one_question + [references[i]]
            log = f"---------------------------------------------------\n{blocks[i]}\n"
            for j in range(len(tmp_reference)):
                if j in range(1, len(tmp_reference) - 1):
                    s = tmp_reference[j].split("\n")
                    ref = s[1]
                    reference_description = s[0]
                elif j == 0:
                    ref = tmp_reference[j]
                    reference_description = "no"
                else:
                    ref = tmp_reference[j]
                    reference_description = "correct"

                # Function Calling
                messages = [
                    {"role": "system", "content": "You are a helpful assistant with access to the following functions. Use them if required - "},
                    {"role": "user", "content": f"你是一名因果信息领域的专家，回答以下{questions[i]}并给出解释。回答长度不超过300字。"},
                    {"role": "function-call", "content": '{"name": "get_causal_information", "arguments": {"data": "Laws"}}'},
                    {"role": "function-response", "content": f"{ref}"},  # a hypothetical response from our function
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,  # adding prompt for generation
                    tools=[get_causal_information],  # our functions (tools)
                )

                terminator_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                ]
                prompt_ids =  tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt').to(model. device)
                generated_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=512,
                    eos_token_id=terminator_ids,
                    bos_token_id=tokenizer.bos_token_id,
                )
                response = tokenizer.decode(generated_ids[0][prompt_ids.shape[-1]:],       skip_special_tokens=False)  # `skip_special_tokens=False` for debug
                response = response.replace("\n", "")
                log += f"{model_path} with {reference_description} reference:\n{response}\n"
            log += "-------------------------------------------------------------------\n\n"
            result_file.write(log)
