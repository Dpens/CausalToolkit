import openai
import pandas as pd

# data loading
with open("./dataset/laws_preschool_education.txt", mode="r", encoding="utf-8") as f:
    blocks = f.read().split("\n\n")
    examples = []
    answers = []
    references = []
    for block in blocks:
        sentences = block.split("\n")
        examples.append(sentences[0])
        answers.append(sentences[2])
        references.append(sentences[3])


api_key = "api key"
model = "gpt-4"

# 用户输入  
# 调用OpenAI的ChatCompletion API  
client = openai.OpenAI(base_url="https://api.openai.com/v1", api_key=api_key)

with open("./result/GPT4_laws_preschool_education.txt", mode="w+", encoding="UTF-8") as result_file:
    for i in range(len(examples)):
        _messages=[
                {"role": "function", "name": "discover_causal_relationship", "content": "无参考信息。"},
                {"role": "user", "content": f"你是一名中华人民共和国的儿童教育方面的法律专家，请根据参考信息，回答以下{examples[i]}，并提供相应的依据，回答长度不超过300字。"} 
            ]  
        response = client.chat.completions.create(  
            model=model,  
            messages=_messages
        )  

        response_wo_reference = response.choices[0].message.content

        _messages=[
                {"role": "function", "name": "discover_causal_relationship", "content": f"{references[i][3:]}"},
                {"role": "user", "content": f"你是一名中华人民共和国的儿童教育方面的法律专家，请根据参考信息，回答以下{examples[i]}，并提供相应的依据，回答长度不超过300字。"} 
            ]  
        response = client.chat.completions.create(  
            model=model,  
            messages=_messages
        )  

        response_with_reference = response.choices[0].message.content
        
        response_wo_reference = response_wo_reference.replace("\n", "")
        response_with_reference = response_with_reference.replace("\n", "")

        # _messages=[
        #         {"role": "user", "content": f"回答1: {response_wo_reference}\n回答2: {response_with_reference}"},  
        #         {"role": "assistant", "content": f"完整题目和答案: {blocks[i]}"},
        #         {"role": "user", "content": "你是一个专业的法律助理，请根据完整题目和答案判断回答1和回答2哪个更准确？解释的更好？"} 
        #     ] 
        # response = client.chat.completions.create(  
        #     model=model,  
        #     messages=_messages
        # )  
        # print(_messages)
        # response_compare = response.choices[0].message.content
        # log = f"---------------------------------------------------\n{blocks[i]}\nGPT4 without reference:\n{response_wo_reference}\nGPT4 with reference:\n{response_with_reference}\nCompare Result: {response_compare}\n-------------------------------------------------\n\n"
        # print(log)
        log = f"---------------------------------------------------\n{blocks[i]}\nGPT4 without reference:\n{response_wo_reference}\nGPT4 with reference:\n{response_with_reference}\n-------------------------------------------------\n\n"
        print(log)
        result_file.write(log)