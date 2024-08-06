import re
import streamlit as st

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from PIL import Image
from io import BytesIO
import base64


processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_document(image):
    # prepare encoder inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
          
    # generate answer
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # postprocess
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    
    return processor.token2json(sequence)

# Streamlit
uploaded_file = st.file_uploader(label="Add Receipt", type=['png','jpg','jpeg'], accept_multiple_files=False, key=None, help="Receipt image uploader", on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

if uploaded_file is not None:
    img = Image.open(uploaded_file)  
    st.write(process_document(img))

base64txt = st.text_area(label="Input Base64 Image", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")

if st.button("Process Image"):
    img_64 = Image.open(BytesIO(base64.b64decode(base64txt)))
    st.write(process_document(img_64))