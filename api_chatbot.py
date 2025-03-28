from transformers import pipeline
import gradio as gr

# 使用 Hugging Face hosted API 模型（instruction-follow）
chatbot = pipeline("text2text-generation", model="google/flan-t5-base")

def respond(message):
    result = chatbot(message, max_length=100)[0]["generated_text"]
    return result

gr.Interface(fn=respond, inputs="text", outputs="text", title="Flan-T5 小型智慧 Chatbot").launch()
