import openai
import pandas as pd

# data loading
with open("./dataset/sachs.txt", mode="r", encoding="utf-8") as f:
    blocks = f.read().split("\n\n")
    examples = []
    answers = []
    references = []
    questions = []
    for block in blocks:
        sentences = block.split("\n")
        examples.append(sentences[0])
        answers.append(sentences[3])
        references.append(sentences[2])
        questions.append(sentences[4])


api_key = "api key"
model = "gpt-4"

# 用户输入  
# 调用OpenAI的ChatCompletion API  
client = openai.OpenAI(base_url="https://api.openai.com/v1", api_key=api_key)

with open("./result/GPT4_sachs.txt", mode="w+", encoding="UTF-8") as result_file:
    for i in range(len(examples)):
        _messages=[
                {"role": "user", "content": examples[i]},  
                {"role": "function", "name": "discover_causal_relationship", "content": "无参考信息。"},
                {"role": "user", "content": f"你是一个专业的生物信息学研究者，请根据上述背景和参考信息，回答以下问题：{questions[i]}回答长度不超过300字。"} 
            ]  
        response = client.chat.completions.create(  
            model=model,  
            messages=_messages
        )  

        response_wo_reference = response.choices[0].message.content

        _messages=[
                {"role": "user", "content": examples[i]},  
                {"role": "function", "name": "discover_causal_relationship", "content": f"{references[i]}"},
                {"role": "user", "content": f"你是一个专业的生物信息学研究者，请根据上述背景和参考信息，回答以下问题：{questions[i]}回答长度不超过300字。"} 
            ]  
        response = client.chat.completions.create(  
            model=model,  
            messages=_messages
        )  

        response_with_reference = response.choices[0].message.content
        
        response_wo_reference = response_wo_reference.replace("\n", "")
        response_with_reference = response_with_reference.replace("\n", "")

        log = f"---------------------------------------------------\n{blocks[i]}\nGPT4 without reference:\n{response_wo_reference}\nGPT4 with reference:\n{response_with_reference}\n-------------------------------------------------\n\n"
        print(log)
        result_file.write(log)