import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

# load datset
data = load_dataset("samsum", split="train")
data_df = data.to_pandas()

# adding prompt for model finetuning
data_df["prompt"] = data_df[["dialogue", "summary"]].apply(lambda x: "###Human: Summarize this following dialogue: " + x["dialogue"] + "\n###Assistant: " +x["summary"], axis=1)

# loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ")

# adding pad token to tokenizer
tokenizer.pad_token = tokenizer.eos_token

# # creating cofig for model quantization
# gptq_config = GPTQConfig(bits=4, disable_exllama=True, tokenizer=tokenizer, exllama_config={"version":2})

# # load quantize model 
# quantized_model = AutoModelForCausalLM.from_pretrained(
#                           "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
#                           quantization_config=gptq_config,
#                           device_map="auto")

# creating bit and bytes cofig for model quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use fp16 for computation
    bnb_4bit_use_double_quant=True,  # Use double quantization for memory efficiency
)

# Load model and tokenizer
quantized_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config=bnb_config,
    device_map="auto",  # Automatically assigns layers to available GPUs
)


# peft
# prepare model before training using peft
quantized_model = prepare_model_for_kbit_training(quantized_model)

# setting the peft config for quantize model fine tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)


quantized_model = get_peft_model(quantized_model, peft_config)

# set training arhuments
training_arguments = TrainingArguments(
        output_dir="mistral-finetuned-samsum",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=1,
        max_steps=250,
        fp16=True,
        push_to_hub=True,
        report_to="none"
  )


# Create the SFTTrainer
trainer = SFTTrainer(
        model=quantized_model,
        train_dataset=data,
        peft_config=peft_config,
        args=training_arguments,
        tokenizer=tokenizer,
    )

# Train model
trainer.train()