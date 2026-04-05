O Sabia-7B Instruction é um modelo de linguagem de grande escala para português, que foi ajustado usando o conjunto de dados de seguimento de instruções [Canarim-Instruct-PTBR](https://huggingface.co/datasets/dominguesm/Canarim-Instruct-PTBR-Dataset).

Este projeto é o trabalho final da disciplina de DeepLearning

O modelo base foi desenvolvido pela Maritaca AI. Veja o [base model](https://huggingface.co/maritaca-ai/sabia-7b).

## Detalhes do treinamento

Devido a limitações de recursos e tempo, a instrução Sabia-7B foi treinada com apenas 10% do conjunto de dados Canarim durante 3 épocas, utilizando a GPU A100 do Google Colab Pro.

Além disso, utilizamos o Ajuste Fino Eficiente de Parâmetros (PEFT) para "congelar" alguns pesos do modelo base e a Adaptação de Classificação Local Quantizada (QLoRA) para reduzir o uso de memória da GPU.

## Exemplo de instrução

```python
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Set bitsandbytes config fow low GPU RAM usage
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
      )

# Load base model
base_model = LlamaForCausalLM.from_pretrained(
        "maritaca-ai/sabia-7b",
        device_map= {"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

# Load fine tuned model
# requires !pip install peft
eval_model = PeftModel.from_pretrained(model, '../Model_pretrained/', device_map = {"": 0}, is_trainable=False)

def gen(prompt, model):
    INTRO_INPUT = "Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que conclua adequadamente a solicitação."
    prompt = INTRO_INPUT + "\n\nInstrução: " + prompt + "\n\nResposta:"
    
    input_ids = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
      output = model.generate(
        input_ids["input_ids"].to(device),
        max_length=1024,
        eos_token_id=tokenizer.encode("</s>")
      )

    #print(output)

    output = output[0][len(input_ids["input_ids"][0]):]

    return tokenizer.decode(output, skip_special_tokens=True)

print gen("O que é inteligência artificial?", eval_model)
```
