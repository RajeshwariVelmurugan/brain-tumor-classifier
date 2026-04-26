
import os
import gradio as gr
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import google.generativeai as genai
from gtts import gTTS
import tempfile

# ---------- SECURITY: Load API key from environment (never hardcode) ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Fallback for local development (do not commit this line)
    GEMINI_API_KEY = input("Enter your Gemini API key: ")

genai.configure(api_key=GEMINI_API_KEY, transport='rest')
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ---------- Load model and class names ----------
model = load_model('brain_tumor_final.keras')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# ---------- Text-to-Speech ----------
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_file.name)
    return temp_file.name

# ---------- Gemini Chatbot with Voice ----------
def gemini_chatbot_with_audio(message, history):
    if not message.strip():
        return "Please ask a question.", None
    chat = gemini_model.start_chat(history=[])
    for turn in history:
        if turn['role'] == 'user':
            chat.send_message(turn['content'])
    response = chat.send_message(message)
    answer = response.text
    full_answer = f"{answer}\n\n⚠️ *Disclaimer: AI-generated – always consult a doctor.*"
    audio_file = text_to_speech(answer)
    return full_answer, audio_file

# ---------- MRI Classifier ----------
def predict(image):
    if image is None:
        return "No image uploaded", None
    img = tf.image.resize(image, (224, 224))
    img_array = np.expand_dims(img.numpy(), axis=0)
    img_array = preprocess_input(img_array)
    pred_probs = model.predict(img_array, verbose=0)[0]
    pred_class = class_names[np.argmax(pred_probs)]
    confidence = np.max(pred_probs) * 100
    probs_dict = {name: float(prob) for name, prob in zip(class_names, pred_probs)}
    return f"**Prediction: {pred_class}**\nConfidence: {confidence:.1f}%", probs_dict

# ---------- Gradio UI ----------
with gr.Blocks(title="Brain Tumor Classifier + AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Brain Tumor Classifier & AI Medical Assistant")
    with gr.Tabs():
        with gr.TabItem("📷 MRI Classifier"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="numpy", label="Upload MRI")
                    submit_btn = gr.Button("Analyze")
                with gr.Column():
                    output_label = gr.Label(label="Prediction")
                    output_probs = gr.Label(label="Probability Distribution")
            submit_btn.click(predict, inputs=input_image, outputs=[output_label, output_probs])
        
        with gr.TabItem("💬 Chatbot + Voice"):
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox(label="Ask anything about brain tumors")
            audio_output = gr.Audio(label="Listen to answer")
            clear = gr.Button("Clear")
            
            def respond(msg, hist):
                txt, aud = gemini_chatbot_with_audio(msg, hist)
                hist.append({"role": "user", "content": msg})
                hist.append({"role": "assistant", "content": txt})
                return hist, aud, ""
            
            msg.submit(respond, [msg, chatbot], [chatbot, audio_output, msg])
            clear.click(lambda: ([], None, ""), None, [chatbot, audio_output, msg])
    
    gr.Markdown("---\n⚠️ **Disclaimer:** For educational use only. Not a medical device.")

demo.launch(share=True)
