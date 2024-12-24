from pydantic import BaseModel


class BotRequest(BaseModel):
    passengerId: str
    input_msg: str