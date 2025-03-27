from transformers import pipeline
import gradio as gr

# 建立聊天模型（用的是免費的對話模型）
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

# 聊天記憶
chat_history = []

def respond(message):
    global chat_history
    from transformers import Conversation
    conversation = Conversation(message, past_user_inputs=[m['user'] for m in chat_history],
                                           generated_responses=[m['bot'] for m in chat_history])
    result = chatbot(conversation)
    reply = result.generated_responses[-1]
    chat_history.append({'user': message, 'bot': reply})
    return reply

# 做成一個簡單的聊天介面
demo = gr.Interface(fn=respond, inputs="text", outputs="text", title="你的 Chatbot", theme="soft")

# 啟動介面
demo.launch()
