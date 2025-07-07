from pydantic import BaseModel, EmailStr

# What user sends during signup
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str

# What user sends during login
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# What we send back after login
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
