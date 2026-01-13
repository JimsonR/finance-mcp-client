"""
Redis client utility for chat persistence.
Establishes connection to Redis Cloud instance using environment variables.
"""
import os
import sys

# Add project root to path to ensure we can load .env if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis
from dotenv import load_dotenv

load_dotenv()


def get_redis_client():
    """
    Establishes and returns a connection to the Redis Cloud instance.
    Returns None if connection fails.
    
    Required environment variables:
    - REDIS_HOST: Redis server hostname
    - REDIS_PORT: Redis server port
    - REDIS_PASSWORD: Redis authentication password
    """
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
        )
        # Fast ping to verify connection
        client.ping()
        return client
    except Exception as e:
        print(f"❌ Redis Connection Error: {e}")
        return None
