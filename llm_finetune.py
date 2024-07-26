import json
import torch
import numpy as np
import transformers
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling
from typing import Any, Dict, List, Union


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        
        response_token_ids = self.tokenizer.encode("### Response(응답):")

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch
    
    

def load_dataset_from_json(json_file_path, test_size=0.2, random_state=42):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 데이터 분할
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # Dataset 객체 생성
    train_dataset = Dataset.from_dict({"instruction": [item["instruction"] for item in train_data],
                                       "input": [item["input"] for item in train_data],
                                       "output": [item["output"] for item in train_data]})

    val_dataset = Dataset.from_dict({"instruction": [item["instruction"] for item in val_data],
                                     "input": [item["input"] for item in val_data],
                                     "output": [item["output"] for item in val_data]})

    return train_dataset, val_dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def gen(x):
    a = PROMPT_DICT['prompt_input'].format(instruction=x, input='(-224, 0, 515)')
    input_ids = tokenizer.encode(a, return_tensors="pt")

    gened = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        num_return_sequences=1,
        early_stopping=True,
        do_sample=False,
        eos_token_id=2,
    )
    response = tokenizer.decode(gened[0])
    return response.split('### Response(응답):')[1].strip()


PROMPT_DICT = {
    "prompt_input": (
        "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    )
}

# JSON 파일 경로
json_file_path = "./data_0601.json"


# train 및 validation 데이터 로드
train_data, val_data = load_dataset_from_json(json_file_path)


# 데이터 출력
print("Train dataset:")
print(train_data)

print("\nValidation dataset:")
print(val_data)

# data
tr_data = train_data.map(
    lambda x:
    {'text': f"### Instruction(명령어):\n{x['instruction']}\n\n### Input(입력):\n{x['input']}\n\n### Response(응답):{x['output']}<|endoftext|>" }
)

#양자화
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")

model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B", quantization_config=bnb_config, device_map={"":0})


tr_data = tr_data.map(lambda samples: tokenizer(samples["text"]), batched=True)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

#LoRA
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )


#train
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tr_data,
    args=transformers.TrainingArguments(
        output_dir = "./",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        save_steps = 1000,
        num_train_epochs=10,
        learning_rate=2e-4,
        fp16=True,
        save_strategy = "steps",
        logging_steps=10,
        optim="paged_adamw_8bit"
    ),
    data_collator=data_collator,
)



model.config.use_cache = False  
trainer.train()


model.eval()
model.config.use_cache = True 

#지시문을 입력
gen("오른쪽으로 빠르게 움직여서 촬영해줘")