from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from app.schemas.auth_schema import SignupRequest, TokenResponse
from app.services import user
from app.core.security import create_jwt_token

auth_router = APIRouter()

@auth_router.post("/signup")
async def signup(payload: SignupRequest):
    try:
        new_user = await user.register_user(payload)
        return {"message": f"{new_user.email} signed up successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@auth_router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    existing_user = await user.login_user(form_data.username, form_data.password)
    if not existing_user:
        raise HTTPException(status_code=401, detail="Wrong email or password")
    
    token = create_jwt_token({"sub": existing_user["email"]})
    return {"access_token": token}
