from app.core.config import db
from app.core.security import hash_password, verify_password
from app.models.user_model import User

# Find user by email
async def find_user(email: str):
    return await db.users.find_one({"email": email})

# Add user to DB
async def register_user(user_data):
    existing = await find_user(user_data.email)
    if existing:
        raise ValueError("Email already in use")

    user = User(
        email=user_data.email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        hashed_password=hash_password(user_data.password)
    )
    await db.users.insert_one(user.to_dict())
    return user

# Authenticate user during login
async def login_user(email: str, password: str):
    user = await find_user(email)
    if user and verify_password(password, user["hashed_password"]):
        return user
    return None
