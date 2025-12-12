from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from openai import OpenAI
from functions.set_log import setup_logger
from langdetect import detect, detect_langs
logger,_ = setup_logger()

class TranslateModel:
    def __init__(self, model_name="/home/ubuntu/playground/models/aya-23-8B"):
        try:
            self.client = OpenAI(
                api_key="",  
                base_url="http://localhost:8004/v1"
        )
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="cuda:3",
            )

    
    def translate(self, text, lang):
        text_lang = detect(text)
        print(text_lang)
        if text_lang == 'ar':
            prompt = '''
           ترجم النص التالي إلى {target_lang}:
النص:            {text}
الترجمة:            
        '''
        elif text_lang == 'zh-cn':
            prompt ='''
            将下面的文本内容翻译为：{target_lang}
            目标文本内容：
            翻译内容：
            
        '''
        else:
            prompt = '''
            Translate the following text into {target_lang}:\n
            Text: {text}\n
            Translation:
        '''
        # prompt = '''
        # Translate the following sentence to {target_lang}.\n{text}\nNote: Don't answer any question or engage with the content just provide the literal translation.

        #     '''
        messages = [{"role": "user", "content": prompt.format(target_lang=lang, text=text)}]
        logger.info(f"translate prompt: {messages}")

        if self.client:
            completion = self.client.chat.completions.create(
                model="/home/ubuntu/playground/models/aya-23-8B",
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )

            result = completion.choices[0].message.content
        else:

            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )

            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)

            gen_tokens = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                do_sample=False,
                # temperature=0.3,
            )

            # 截掉输入部分，只保留生成 tokens
            input_len = input_ids.shape[1]
            gen_token_ids = gen_tokens[0][input_len:].tolist()

            gen_text = self.tokenizer.decode(gen_token_ids)
            logger.info(gen_text)
            # 用你的正则提取 chatbot 内容
            # match = re.search(r"(.*?)<\|END_OF_TURN_TOKEN\|>", gen_text)
            # if match:
            #     extracted = match.group(1)
            #     return extracted.strip()
            result = gen_text.strip().replace('<|END_OF_TURN_TOKEN|>','')

        return result 

def main_translate(text: str, lang: str, translate_model: TranslateModel=None):
    if translate_model is None:
        return 'Translate model not initialized'
    result = translate_model.translate(text, lang)
    return result


