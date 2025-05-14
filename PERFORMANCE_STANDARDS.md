# Forest OS Performance & Integrity Standards

## Performance Targets

### Latency
- **P75 Latency Target**: 500ms for core API endpoints
- **Maximum Acceptable Latency**: 1000ms
- **Error Budget**: 0.1% of total requests

### Monitoring Strategy
- Use Sentry for performance tracking
- Log all API request timings
- Track LLM call performance and token usage

## Transactional Consistency

### Database Transactions
- Default Isolation Level: REPEATABLE READ
- Use SQLAlchemy's `session.begin()` for explicit transaction management
- Implement retry mechanisms for transient database errors

## Audit Trail Structure

### Basic Audit Log Entry
```python
class AuditLogEntry:
    timestamp_utc: datetime
    user_id: UUID
    action: str
    resource_type: str
    resource_id: UUID
    metadata: dict
```

## API Idempotency Considerations

### Idempotency Key Strategy
- Generate unique idempotency keys for critical write operations
- Store and validate idempotency keys to prevent duplicate processing
- Implement a time-based expiration for idempotency keys

## Error Handling Principles
- Provide clear, actionable error messages
- Log all errors with sufficient context
- Implement graceful degradation for non-critical service failures

## Continuous Improvement
- Regularly review performance metrics
- Conduct periodic load testing
- Update performance targets based on real-world usage data
