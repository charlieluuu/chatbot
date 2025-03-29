import requests
import gradio as gr
import json

chat_history = []

def chat_with_ollama(message, history):
    global chat_history
    chat_history.append({"role": "user", "content": message})

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama2", "messages": chat_history}
        )

        full_response = ""
        for line in response.text.strip().splitlines():
            try:
                result = json.loads(line)
                if "message" in result and "content" in result["message"]:
                    full_response += result["message"]["content"]
            except:
                continue

        chat_history.append({"role": "assistant", "content": full_response})
        history.append((message, full_response))
        return history, history

    except Exception as e:
        history.append((message, f"[錯誤] {str(e)}"))
        return history, history

def reset_chat():
    global chat_history
    chat_history = []
    return []

with gr.Blocks() as demo:
    gr.Markdown("## 🦙 local LLaMA Chatbot")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="message input", placeholder="Type your message here...")
    btn = gr.Button("sent")
    clear = gr.Button("clear chat")

    btn.click(fn=chat_with_ollama, inputs=[msg, chatbot], outputs=[chatbot, chatbot])
    msg.submit(fn=chat_with_ollama, inputs=[msg, chatbot], outputs=[chatbot, chatbot])
    clear.click(fn=reset_chat, outputs=[chatbot])

demo.launch()
