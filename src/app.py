from fastapi import FastAPI
from chainlit.utils import mount_chainlit

app = FastAPI()

@app.get("/main")
def read_main():
    return {"message": "CMO Chainlit API"}

mount_chainlit(app=app, target="main.py", path="/chat")
