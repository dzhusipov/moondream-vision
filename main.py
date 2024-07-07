from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io

app = FastAPI()

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

@app.post("/describe_image/")
async def describe_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    enc_image = model.encode_image(image)
    description = model.answer_question(enc_image, "Describe this image.", tokenizer)
    return JSONResponse(content={"description": description})

# Run the app using the command: uvicorn filename:app --reload