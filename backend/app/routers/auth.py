from fastapi import APIRouter, HTTPException, status
from app.auth import authenticate_user, create_access_token, decode_token

router = APIRouter()

@router.post("/login")
async def login(payload: dict):
    username = payload.get("username", "")
    password = payload.get("password", "")

    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    token = create_access_token({
        "sub": user["username"],
        "role": user["role"],
        "full_name": user["full_name"]
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"],
        "full_name": user["full_name"]
    }

@router.get("/me")
async def get_me(token: str):
    payload = decode_token(token)
    return {
        "username": payload.get("sub"),
        "role": payload.get("role"),
        "full_name": payload.get("full_name")
    }
