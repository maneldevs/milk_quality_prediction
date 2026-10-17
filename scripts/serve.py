import uvicorn

from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
from predict import PredictionResponse, process_prediction


app = FastAPI(title="Milk Quality Prediction")
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

class MilkRequest(BaseModel):
    ph: float = Field(ge=3.0, le=9.0)
    temperature: float = Field(ge=0.0, le=100.0)
    taste: int = Field(ge=0, le=1)
    odor: int = Field(ge=0, le=1)
    fat: int = Field(ge=0, le=1)
    turbidity: int = Field(ge=0, le=1)
    colour: int = Field(ge=240, le=255)

@app.post("/api/predict", response_model=PredictionResponse)
def predict_api(milk: MilkRequest):
    return process_prediction(Path("models/modelXB.bin"), milk)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"prediction": None, "is_high_grade": False})

@app.post("/", response_class=HTMLResponse)
def predict(
        request: Request,
        ph: float = Form(...),
        temperature: float = Form(...),
        taste: int = Form(...),
        odor: int = Form(...),
        fat: int = Form(...),
        turbidity: int = Form(...),
        colour: int = Form(...)
    ):
    try:
        data = {
            "ph": ph, "temperature": temperature, "taste": taste,
            "odor": odor, "fat": fat, "turbidity": turbidity, "colour": colour
        }
        milk = MilkRequest.model_validate(data)
    except ValidationError as e:
        errors = e.errors()
        first_error = errors[0]
        field = first_error["loc"][0]
        msg = first_error["msg"]
        message = f"Field {field} is invalid: {msg}"
        return templates.TemplateResponse(
            request, 
            "index.html", 
            {
                "prediction": None, 
                "is_high_grade": False, 
                "error": message,
                "ph": ph,
                "temperature": temperature,
                "colour": colour,
                "taste": taste,
                "odor": odor,
                "fat": fat,
                "turbidity": turbidity
            })
    
    prediction = process_prediction(Path("models/modelXB.bin"), milk)
    
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "prediction": prediction.prediction,
            "is_high_grade": prediction.is_high_grade,
            "ph": milk.ph,
            "temperature": milk.temperature,
            "colour": milk.colour,
            "taste": milk.taste,
            "odor": milk.odor,
            "fat": milk.fat,
            "turbidity": milk.turbidity
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)