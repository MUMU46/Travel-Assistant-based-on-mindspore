import mindnlp
import os
import sys
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import mindspore as ms
from mindnlp.peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from mindspore.communication import init, get_rank, get_group_size

def train():
    # ================= 1. 分布式环境初始化 =================
    try:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        print(f"🚀 启动分布式训练: Rank {rank_id}/{rank_size}")
    except:
        rank_id = 0
        rank_size = 1
        print("⚠️ 未检测到分布式环境，启动单卡模式")

    # ================= 2. 配置参数 =================
    MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
    DATA_FILE = "./data_preprocess/train_data.jsonl"
    OUTPUT_DIR = "./qwen2.5-7B_lora_output"
    CACHE_DIR = "/cache/hf" 
    
    # ================= 3. 加载模型与Tokenizer =================
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, mirror='modelscope', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

        # ================= 4. 模型加载（4bit + BF16） =================
   
    print("正在加载模型 (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        mirror='modelscope',
        cache_dir=CACHE_DIR,  
        ms_dtype=ms.bfloat16,    # Ascend 910B 推荐使用 BF16
        trust_remote_code=True,
        attn_implementation="eager" # 🌟 关键：强制关闭 FlashAttn，解决 bprop 报错
    )

    # ================= 4. LoRA 配置 =================
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,              # LoRA 秩
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # 针对 Qwen 的关键层
    )
    model = get_peft_model(model, peft_config)
    
    # 🌟 显存优化：开启梯度检查点
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if rank_id == 0:
        model.print_trainable_parameters()

    # ================= 5. 数据处理 =================
    # 加载数据集
    dataset = load_dataset('json', data_files=DATA_FILE, split='train')

    def process_func(examples):
        """将数据转换为 Qwen 的 ChatML 格式 input_ids"""
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
        
        full_texts = []
        for instr, inp, out in zip(instructions, inputs, outputs):
            # 手动构建 ChatML 格式
            text = f"<|im_start|>system\n{instr}<|im_end|>\n<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
            full_texts.append(text)
            
        # Tokenize
        # max_length 设为 1024 或 2048 以节省显存
        tokenized = tokenizer(full_texts, padding="max_length", truncation=True, max_length=1024)
        
        # 将 input_ids 复制给 labels (全量计算 Loss)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = dataset.map(process_func, batched=True)

    # ================= 6. 训练参数 =================
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,   # 14B 模型单卡 Batch 只能设 1
        gradient_accumulation_steps=4,   # 梯度累积，等效 Batch Size = 8 * GPU数
        gradient_checkpointing=True,     # 必须开启
        
        # 🌟 关键：Ascend 910B 必须用 bf16，不能用 fp16 (避开 amp 报错)
        fp16=False,
        bf16=True,
        
        num_train_epochs=3,              # 训练轮数
        learning_rate=2e-4,              # LoRA 学习率
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        report_to=[],                  # 不上传 wandb
        per_device_eval_batch_size=1,  # 评估时也用小batch
        eval_accumulation_steps=1,
        remove_unused_columns=True,  # 自动移除无用列
    )
    

    # ================= 7. 开始训练 =================
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    print("🚀 开始训练...")
    trainer.train()
    
    # 保存权重 (仅主卡)
    if rank_id == 0:
        model.save_pretrained(OUTPUT_DIR)
        print(f"✅ 训练完成！LoRA 权重已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    train()