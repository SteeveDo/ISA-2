from functions import *
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class Item(BaseModel):
    message:str

@app.get("/test")
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


