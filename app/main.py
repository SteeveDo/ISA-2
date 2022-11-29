from functions import *
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app=FastAPI()

class Item(BaseModel):
    message:str

@app.get("/")
async def root():
    return {
            "message": "API CHATBOT"
            }

@app.post("/ticket/{message}")
async def find_solutions(message: str):
    
    create_file(message)
    texts= normalisation_text()

    DataVecs = getAvgFeatureVecs(texts)

    df_result_label=getRecommandations(DataVecs)

    recommandations=get_solutions(df_result_label)

    delete_file()

    return recommandations

@app.post("/message")
async def find_answers(item:Item):

    create_file(item.message)
    texts= normalisation_text()

    DataVecs = getAvgFeatureVecs(texts)

    df_result_label=getRecommandations(DataVecs)

    recommandations=get_solutions(df_result_label)

    delete_file()
    
    return recommandations


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7000)