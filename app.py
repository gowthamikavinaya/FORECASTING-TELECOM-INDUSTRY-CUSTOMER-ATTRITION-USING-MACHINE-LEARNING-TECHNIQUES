from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import XGBoostModel
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
if os.path.exists("/static/result.png"):
  os.remove("/static/result.png")
model = XGBoostModel()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    city_list = model.city_list
    return templates.TemplateResponse("index.html", {"request": request, "context": "Loading", "city": city_list})

@app.post("/predict", response_class=HTMLResponse)
async def get_prediction(request: Request):
    input_dict = await request.form()
    y_pred , img, feedback = model.predict(input_dict)
    prediction = "High Chance of Churn"
    if y_pred == 0:
        prediction = "Low Chance of Churn"
    img.savefig('static/result.png',bbox_inches='tight')
    return templates.TemplateResponse("index.html", {"request": request, "context": prediction, "feedback": feedback})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)