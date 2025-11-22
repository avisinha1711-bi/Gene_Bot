import os

# Deployment environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Database configuration for production
if ENVIRONMENT == 'production':
    DATABASE_URL = os.getenv('DATABASE_URL')
    DEBUG = False
else:
    DATABASE_URL = 'sqlite:///local.db'
    DEBUG = True
