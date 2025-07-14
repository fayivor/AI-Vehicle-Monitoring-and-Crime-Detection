"""Authentication and authorization endpoints."""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Dict, Any

from ...models.user import User, UserCreate, UserLogin, Token, UserSession
from ...utils.config import get_settings
from ...utils.security import create_access_token, verify_token, hash_password, verify_password
from ...utils.logging import get_logger, audit_logger

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current authenticated user from token.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = verify_token(token)
    if token_data is None:
        audit_logger.log_security_event(
            "INVALID_TOKEN", 
            "unknown", 
            f"Invalid token provided: {token[:20]}...",
            "WARNING"
        )
        raise credentials_exception
    
    # TODO: Implement user lookup from database
    # For now, create a mock user based on token data
    user = User(
        user_id=token_data.user_id or "mock-user",
        username=token_data.username or "mock-user",
        email="mock@ghana.gov.gh",
        full_name="Mock User",
        role=token_data.role or "officer",
        security_clearance=token_data.security_clearance or "public",
        is_active=True,
        permissions=["view_vehicles", "view_incidents"]
    )
    
    if not user.is_active:
        audit_logger.log_security_event(
            "INACTIVE_USER_ACCESS", 
            user.user_id, 
            "Inactive user attempted access",
            "WARNING"
        )
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """
    OAuth2 compatible token login endpoint.
    
    Args:
        form_data: Username and password from form
        
    Returns:
        Access token and user information
        
    Raises:
        HTTPException: If authentication fails
    """
    # TODO: Implement actual user authentication against database
    # For now, use mock authentication
    
    # Mock user validation
    if form_data.username == "admin" and form_data.password == "admin123":
        user = User(
            user_id="admin-001",
            username="admin",
            email="admin@ghana.gov.gh",
            full_name="System Administrator",
            role="administrator",
            security_clearance="secret",
            is_active=True,
            permissions=["*"]  # All permissions
        )
    elif form_data.username == "officer" and form_data.password == "officer123":
        user = User(
            user_id="officer-001",
            username="officer",
            email="officer@ghana.gov.gh",
            full_name="Police Officer",
            role="officer",
            security_clearance="restricted",
            is_active=True,
            permissions=["view_vehicles", "view_incidents", "create_incidents"]
        )
    else:
        audit_logger.log_security_event(
            "FAILED_LOGIN", 
            form_data.username, 
            "Invalid credentials provided",
            "WARNING"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.user_id,
            "role": user.role,
            "security_clearance": user.security_clearance,
        },
        expires_delta=access_token_expires
    )
    
    # Log successful login
    audit_logger.log_user_access(
        user.user_id,
        "authentication",
        "login",
        True
    )
    
    logger.info(
        "User logged in successfully",
        user_id=user.user_id,
        username=user.username,
        role=user.role
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        user=user
    )


@router.post("/login", response_model=Token)
async def login(user_login: UserLogin) -> Token:
    """
    Alternative login endpoint with JSON payload.
    
    Args:
        user_login: Login credentials
        
    Returns:
        Access token and user information
    """
    # Use the same logic as token endpoint
    from fastapi.security import OAuth2PasswordRequestForm
    
    # Create form data from JSON
    form_data = OAuth2PasswordRequestForm(
        username=user_login.username,
        password=user_login.password
    )
    
    return await login_for_access_token(form_data)


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user information
    """
    audit_logger.log_user_access(
        current_user.user_id,
        "user_profile",
        "view",
        True
    )
    
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)) -> Dict[str, str]:
    """
    Logout endpoint (token invalidation would be handled client-side or with token blacklist).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Logout confirmation
    """
    audit_logger.log_user_access(
        current_user.user_id,
        "authentication",
        "logout",
        True
    )
    
    logger.info(
        "User logged out",
        user_id=current_user.user_id,
        username=current_user.username
    )
    
    return {"message": "Successfully logged out"}


@router.post("/register", response_model=User)
async def register_user(
    user_create: UserCreate,
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Register a new user (admin only).
    
    Args:
        user_create: User creation data
        current_user: Current authenticated user (must be admin)
        
    Returns:
        Created user information
        
    Raises:
        HTTPException: If user is not authorized or username exists
    """
    # Check if current user is administrator
    if current_user.role != "administrator":
        audit_logger.log_security_event(
            "UNAUTHORIZED_USER_CREATION",
            current_user.user_id,
            f"Non-admin user attempted to create user: {user_create.username}",
            "WARNING"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can create users"
        )
    
    # TODO: Implement actual user creation in database
    # For now, return mock created user
    
    # Hash password
    hashed_password = hash_password(user_create.password)
    
    # Create user (mock)
    new_user = User(
        user_id=f"user-{user_create.username}",
        username=user_create.username,
        email=user_create.email,
        full_name=user_create.full_name,
        role=user_create.role,
        security_clearance=user_create.security_clearance,
        department=user_create.department,
        badge_number=user_create.badge_number,
        phone=user_create.phone,
        is_active=True,
        permissions=user_create.permissions,
        created_by=current_user.user_id
    )
    
    audit_logger.log_user_access(
        current_user.user_id,
        "user_management",
        f"create_user:{new_user.user_id}",
        True
    )
    
    logger.info(
        "New user created",
        created_user_id=new_user.user_id,
        created_by=current_user.user_id,
        role=new_user.role
    )
    
    return new_user
