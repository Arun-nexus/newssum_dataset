from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import gen, device
app = FastAPI(title="kalachakra-newslm")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")

@app.get("/", response_class=HTMLResponse)
def home(req: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": req}
    )
@app.post("/generate")
def generate(year: int, question: str, max_tokens: int = 150):
    prompt = f"[YEAR={year}]\n{question}"
    print(prompt)
    ans = gen(
        prompt=prompt,
        max_tokens=max_tokens
    )

    return {
        "year": year,
        "question": question,
        "answer": ans
    }
