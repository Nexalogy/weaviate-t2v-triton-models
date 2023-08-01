import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from vectorizer import Vectorizer, VectorInput
from meta import Meta


app = FastAPI()
vec : Vectorizer
meta_config : Meta
logger = getLogger('uvicorn')


@app.on_event("startup")
def startup_event():
	global vec
	global meta_config

	modelName = os.environ['MODEL_NAME']
	tritonUrl = os.environ['TRITON_URL']

	meta_config = Meta(modelName, tritonUrl)
	vec = Vectorizer()


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
	response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta():
	return meta_config.get()


@app.post("/vectors")
@app.post("/vectors/")
async def read_item(item: VectorInput, response: Response):
	try:
		vector = await vec.vectorize(item.text, item.config)
		return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
	except Exception as e:
		logger.exception(
			'Something went wrong while vectorizing data.'
		)
		response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
		return {"error": str(e)}
