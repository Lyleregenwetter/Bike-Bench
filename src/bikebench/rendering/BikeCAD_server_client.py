import requests

from bikebench.exceptions import InternalError
from bikebench.rendering.BikeCAD_server_manager import ServerManager


class RenderingClient:

    def __init__(self,
                 server_manager: ServerManager):
        self._server_manager = server_manager

    def render(self, bike_xml: str):
        endpoint = self._server_manager.endpoint("/api/v1/render")
        result = requests.post(endpoint, data=bike_xml)
        if result.status_code == 200:
            return result.content
        raise InternalError(f"Rendering request failed {result}")
