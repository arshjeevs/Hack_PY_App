from pydantic import BaseModel, EmailStr

class User(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    hashed_password: str
    role: str = "user"

    def to_dict(self):
        return self.dict()
