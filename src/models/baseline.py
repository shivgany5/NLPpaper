import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaselineLLM:
    """
    Baseline Zero-shot architecture utilizing a massive LLM (e.g., Llama 3 70B).
    """
    def __init__(self, config):
        model_id = config['models']['baseline']['model_id']
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # We enforce 4-bit config implicitly as large LLMs consume major VRAM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            load_in_4bit=True,
            device_map="auto"
        )

    def generate(self, prompt: str) -> str:
        formatted_prompt = f"Answer the following question about classical literature as comprehensively as possible.\n\nQuestion: {prompt}\n\nAnswer:"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        # Return only the newly generated text
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return generated_text.strip()
