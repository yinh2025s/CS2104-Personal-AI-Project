import gradio as gr
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
print(f"Loading model on: {device_name}...")

generator = pipeline('text-generation', model='gpt2', device=device)

def ai_generate(text_input, max_length):
    try:
        result = generator(text_input, max_length=int(max_length), num_return_sequences=1)
        generated_text = result[0]['generated_text']
        
        status_msg = f"\n\n[System Status]\nProcessed on: {device_name}\nGPU Memory: Optimized"
        return generated_text + status_msg
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Personal AI Server (Powered by RTX 3090)")
    gr.Markdown("Enter a prompt below, and your home server will complete the text for you.")
    
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Input Text", placeholder="Once upon a time in a digital world...", lines=5)
            slider = gr.Slider(minimum=10, maximum=100, value=50, label="Max Length")
            btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            out = gr.Textbox(label="AI Output", lines=8)
            
    btn.click(fn=ai_generate, inputs=[inp, slider], outputs=out)

if __name__ == "__main__":
    demo.launch(share=True)