from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

# 載入模型與 tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

def respond(user_input):
    global chat_history_ids
    # 將使用者輸入轉為 token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # 若已有對話歷史就加上去，否則只有這次輸入
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # 產生回應
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # 解碼回應文字
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Gradio UI
demo = gr.Interface(fn=respond, inputs="text", outputs="text", title="My Chatbot")
demo.launch()
