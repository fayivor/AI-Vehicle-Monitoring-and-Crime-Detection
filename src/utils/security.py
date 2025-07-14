"""Security utilities for authentication and authorization."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

from ..models.user import User, TokenData, UserRole, SecurityClearance
from .config import get_settings

settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role")
        security_clearance: str = payload.get("security_clearance")
        
        if username is None:
            return None
            
        token_data = TokenData(
            username=username,
            user_id=user_id,
            role=UserRole(role) if role else None,
            security_clearance=SecurityClearance(security_clearance) if security_clearance else None
        )
        return token_data
    except JWTError:
        return None


def check_permission(user: User, required_permission: str) -> bool:
    """
    Check if a user has a specific permission.
    
    Args:
        user: User object
        required_permission: Permission to check
        
    Returns:
        True if user has permission, False otherwise
    """
    # Administrators have all permissions
    if user.role == UserRole.ADMINISTRATOR:
        return True
    
    # Check specific permissions
    return required_permission in user.permissions


def check_security_clearance(user: User, required_clearance: SecurityClearance) -> bool:
    """
    Check if a user has sufficient security clearance.
    
    Args:
        user: User object
        required_clearance: Required security clearance level
        
    Returns:
        True if user has sufficient clearance, False otherwise
    """
    clearance_levels = {
        SecurityClearance.PUBLIC: 0,
        SecurityClearance.RESTRICTED: 1,
        SecurityClearance.CONFIDENTIAL: 2,
        SecurityClearance.SECRET: 3,
    }
    
    user_level = clearance_levels.get(user.security_clearance, 0)
    required_level = clearance_levels.get(required_clearance, 0)
    
    return user_level >= required_level


def filter_sensitive_data(data: Dict[str, Any], user: User) -> Dict[str, Any]:
    """
    Filter sensitive data based on user's security clearance.
    
    Args:
        data: Data to filter
        user: User requesting the data
        
    Returns:
        Filtered data based on security clearance
    """
    filtered_data = data.copy()
    
    # Define sensitive fields by clearance level
    sensitive_fields = {
        SecurityClearance.RESTRICTED: ["owner_details.id_number", "owner_details.phone"],
        SecurityClearance.CONFIDENTIAL: ["investigation_notes", "witnesses"],
        SecurityClearance.SECRET: ["evidence_files", "related_incidents"]
    }
    
    # Remove fields that require higher clearance
    for clearance, fields in sensitive_fields.items():
        if not check_security_clearance(user, clearance):
            for field in fields:
                # Handle nested field removal
                if "." in field:
                    parts = field.split(".")
                    current = filtered_data
                    for part in parts[:-1]:
                        if part in current and isinstance(current[part], dict):
                            current = current[part]
                        else:
                            break
                    else:
                        if parts[-1] in current:
                            current[parts[-1]] = "[CLASSIFIED]"
                else:
                    if field in filtered_data:
                        filtered_data[field] = "[CLASSIFIED]"
    
    return filtered_data


def encrypt_sensitive_field(value: str) -> str:
    """
    Encrypt sensitive field values for storage.
    
    Args:
        value: Value to encrypt
        
    Returns:
        Encrypted value
    """
    # This is a placeholder - implement proper encryption
    # using the encryption_key from settings
    from cryptography.fernet import Fernet
    
    # In production, use settings.encryption_key
    # For now, generate a key (this should be stored securely)
    key = Fernet.generate_key()
    f = Fernet(key)
    
    encrypted_value = f.encrypt(value.encode())
    return encrypted_value.decode()


def decrypt_sensitive_field(encrypted_value: str) -> str:
    """
    Decrypt sensitive field values.
    
    Args:
        encrypted_value: Encrypted value
        
    Returns:
        Decrypted value
    """
    # This is a placeholder - implement proper decryption
    # using the encryption_key from settings
    from cryptography.fernet import Fernet
    
    # In production, use settings.encryption_key
    # This is just for demonstration
    try:
        # This would fail in practice since we don't have the actual key
        # Implement proper key management
        return encrypted_value  # Return as-is for now
    except Exception:
        return "[DECRYPTION_ERROR]"


class SecurityValidator:
    """Security validation utilities."""
    
    @staticmethod
    def validate_ghana_id(id_number: str) -> bool:
        """
        Validate Ghana national ID format.
        
        Args:
            id_number: ID number to validate
            
        Returns:
            True if valid format, False otherwise
        """
        # Ghana ID format: GHA followed by 9 digits
        import re
        pattern = r"^GHA\d{9}$"
        return bool(re.match(pattern, id_number))
    
    @staticmethod
    def validate_ghana_phone(phone: str) -> bool:
        """
        Validate Ghana phone number format.
        
        Args:
            phone: Phone number to validate
            
        Returns:
            True if valid format, False otherwise
        """
        # Ghana phone format: +233 followed by 9 digits
        import re
        pattern = r"^\+233\d{9}$"
        return bool(re.match(pattern, phone))
    
    @staticmethod
    def validate_vehicle_registration(reg_number: str) -> bool:
        """
        Validate Ghana vehicle registration format.
        
        Args:
            reg_number: Registration number to validate
            
        Returns:
            True if valid format, False otherwise
        """
        # Ghana vehicle registration format: GH-XXXX-YY
        import re
        pattern = r"^GH-\d{4}-\d{2}$"
        return bool(re.match(pattern, reg_number))
