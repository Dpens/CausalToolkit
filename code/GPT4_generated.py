import openai
import pandas as pd


api_key = "api key"
model = "gpt-4"

# 用户输入  
# 调用OpenAI的ChatCompletion API  
client = openai.OpenAI(base_url="https://api.openai.com/v1", api_key=api_key)

for dataset in ["linear", "polynomial"]:
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

    with open(f"./result/GPT4_{dataset}_generated.txt", mode="w+", encoding="UTF-8") as result_file:
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
                messages=[
                        {"role": "function", "name": "discover_causal_relationship", "content": f"{ref}"},
                        {"role": "user", "content": f"你是一个专业的生物信息学研究者，请根据上述参考信息，回答{questions[i]}回答长度不超过300字。"} 
                    ]  
                response = client.chat.completions.create(  
                    model=model,  
                    messages=messages
                )  
                response = response.choices[0].message.content
                response = response.replace("\n", "")
                log += f"GPT4 with {reference_description} reference:\n{response}\n"
            log += "-------------------------------------------------------------------\n\n"
            result_file.write(log)