# CausalToolkit
A toolkit based on Function Calling to enhance the causal reasoning ability of LLMs

## Usage

To simplify the experimental process, we provide a method to simulate the causal knowledge injection process of Function Calling, bypassing the step of invoking CasualToolkit and instead directly providing the causal knowledge that CasualToolkit would have returned. This approach can significantly reduce time costs, allowing for rapid reproduction of experimental results. You can reproduce the results by executing the corresponding files, for example:
```
python GPT4_laws.py
```

In addition, I have also provided a complete experiment framework for invoking CausalToolkit. First, you need to deploy the causal toolkit locally:


```
uvicorn CausalScript:app --reload --port 8001
```

Then, execute the following command:
```
python usage_casualtoolkit.py
```

## Resources

Gemma2-9b-function-calling: <https://huggingface.co/DiTy/gemma-2-9b-it-function-calling-GGUF>

Llama3-8b-function-calling: <https://huggingface.co/Trelis/Meta-Llama-3-8B-Instruct-function-calling>
