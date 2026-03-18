import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class FineTunedLLM:
    """
    Adapter fine-tuned approach utilizing 8B base parameters + trained LoRA parameters.
    """
    def __init__(self, config):
        base_model_id = config['models']['fine_tuned']['base_model_id']
        adapter_path = config['models']['fine_tuned']['adapter_path']
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        # Load the base 8B LLM
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            load_in_4bit=True,
            device_map="auto"
        )
        
        # Attach the domain-specific LoRA adapters (trained on 500 instruction pairs)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)

    def generate(self, prompt: str) -> str:
        # Utilizing standard instruction formatting assumed from tuning
        formatted_prompt = f"Instruction: {prompt}\nResponse:"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return generated_text.strip()
