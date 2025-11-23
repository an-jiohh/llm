# api_app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from model_util import model, tokenizer, generate_text

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class LottoRequest(BaseModel):
    money: int
    winning: str  # "1,2,3,4,5,6"
    bonus: int

# HTML 페이지
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/lotto")
def lotto(req: LottoRequest):
    prompt = (
        f"money={req.money}\n"
        f"winning={req.winning}\n"
        f"bonus={req.bonus}\n"
        "###\n"
    )
    text = generate_text(model, tokenizer, prompt, device="cpu")
    return {"result": text}