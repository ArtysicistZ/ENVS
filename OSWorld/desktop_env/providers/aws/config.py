import os


# Default TTL minutes for instance auto-termination (cloud-side scheduler)
# Can be overridden via environment variable DEFAULT_TTL_MINUTES
DEFAULT_TTL_MINUTES: int = int(os.getenv("DEFAULT_TTL_MINUTES", "180"))

# Master switch for TTL feature
ENABLE_TTL: bool = os.getenv("ENABLE_TTL", "true").lower() == "true"

# EventBridge Scheduler role ARN for scheduling EC2 termination
AWS_SCHEDULER_ROLE_ARN: str = os.getenv("AWS_SCHEDULER_ROLE_ARN", "").strip()

# Root-owned reset daemon inside the VM.
AWS_RESETD_PORT: int = int(os.getenv("AWS_RESETD_PORT", "5001"))
AWS_RESETD_REQUEST_TIMEOUT: int = int(os.getenv("AWS_RESETD_REQUEST_TIMEOUT", "120"))


def compute_ttl_seconds(ttl_minutes: int) -> int:
    try:
        return max(0, int(ttl_minutes) * 60)
    except Exception:
        return 0

