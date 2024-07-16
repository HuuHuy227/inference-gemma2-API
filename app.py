import os
from contextlib import asynccontextmanager
from typing import Optional

from typing_extensions import Annotated

from chat_model import ChatModel
from utils import torch_gc

from chat import (
    create_chat_completion_response,
    create_stream_chat_completion_response,
)
from type import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList
)

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from sse_starlette import EventSourceResponse
import uvicorn


@asynccontextmanager
async def lifespan(app: "FastAPI"):  # collects GPU memory
    yield
    torch_gc()


def create_app(chat_model: "ChatModel") -> "FastAPI":
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Uncomment for security API key
    # api_key = os.environ.get("API_KEY")
    # security = HTTPBearer(auto_error=False)

    # async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
    #     if api_key and (auth is None or auth.credentials != api_key):
    #         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    @app.get("/")
    def read_root():
        return {"Hello": "This is gemma2 inference API"}

    @app.get(
        "/models",
        response_model=ModelList,
        status_code=status.HTTP_200_OK,
        # dependencies=[Depends(verify_api_key)],
    )
    async def list_models():
        model_card = ModelCard(id="gemma2-vn", owned_by= "vnpttgg")
        return ModelList(data=[model_card])

    @app.post(
        "/chat/completions",
        response_model=ChatCompletionResponse,
        status_code=status.HTTP_200_OK,
        # dependencies=[Depends(verify_api_key)],
    )
    async def create_chat_completion(request: ChatCompletionRequest):
        # if not chat_model.engine.can_generate:
        #     raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        if request.stream:
            generate = create_stream_chat_completion_response(request, chat_model)
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            return await create_chat_completion_response(request, chat_model)

    # @app.post(
    #     "/v1/score/evaluation",
    #     response_model=ScoreEvaluationResponse,
    #     status_code=status.HTTP_200_OK,
    #     dependencies=[Depends(verify_api_key)],
    # )
    # async def create_score_evaluation(request: ScoreEvaluationRequest):
    #     if chat_model.engine.can_generate:
    #         raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

    #     return await create_score_evaluation_response(request, chat_model)

    # return app

def run_api() -> None:
    model_path = "Huy227/gemma2_vn"
    generate_config = {
                "max_new_tokens": 1024,
                "top_p":0.95,
                "top_k":40,
                "temperature":0.3,  
                "do_sample": True
            }
    chat_model = ChatModel(model_path, generate_config)
    app = create_app(chat_model)
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8000"))
    print("Visit http://localhost:{}/docs for API document.".format(api_port))
    uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    run_api()