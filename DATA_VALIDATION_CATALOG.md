# Forest OS Data Validation Rules Catalog

## Core Principles
1. Validate at entry point
2. Use Pydantic for type and constraint validation
3. Sanitize and normalize inputs
4. Implement comprehensive error handling

## User Data Validation

### User Registration
- Email:
  - Must be valid email format
  - Unique across system
  - Max length: 255 characters
- Password:
  - Minimum 12 characters
  - Must include uppercase, lowercase, number, special character
  - No common passwords allowed

### Goal/Task Input
- Title:
  - 5-200 characters
  - No special characters except spaces, hyphens, apostrophes
- Description:
  - Optional
  - Max 1000 characters
- Priority:
  - Enum: [critical, high, medium, low]

## System Constraints

### HTA Tree Validation
- Maximum depth: 10 levels
- Maximum nodes per tree: 500
- Node title: 5-100 characters
- Status: [pending, in-progress, completed, blocked]

### Memory Snapshot
- Maximum snapshots per user: 100
- Snapshot size limit: 10 KB
- Retention period: 1 year

## Performance Validation
- API response time: < 500ms
- Error rate: < 0.1%
- Database query time: < 100ms

## Security Validation
- Input sanitization
- No SQL injection
- No XSS vulnerabilities
- Rate limiting
- JWT token validation

## Monitoring & Logging
- Log all validation failures
- Capture detailed error context
- Anonymize sensitive information

## Recommended Validation Libraries
- Pydantic
- email-validator
- password-strength
- bleach (for HTML sanitization)

## Example Validation Pattern
```python
from pydantic import BaseModel, EmailStr, validator

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    
    @validator('password')
    def validate_password_strength(cls, v):
        # Complex password validation logic
        pass
```
