# salespath_env/server/app.py

from openenv.core.env_server import create_fastapi_app

from ..models import (
    SalesPathAction,
    SalesPathObservation,
)
from .salespath_environment import (
    SalesPathEnvironment,
)


app = create_fastapi_app(
    SalesPathEnvironment,
    SalesPathAction,
    SalesPathObservation,
)