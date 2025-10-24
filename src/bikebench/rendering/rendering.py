import attrs

from bikebench.rendering.BikeCAD_server_client import RenderingClient
from bikebench.rendering.BikeCAD_server_manager import SingleThreadedBikeCadServerManager, \
    MultiThreadedBikeCadServerManager
from bikebench.resource_utils import STANDARD_BIKE_RESOURCE
from bikebench.xml_handling.cad_builder import BikeCadFileBuilder

FILE_BUILDER = BikeCadFileBuilder()


@attrs.define(frozen=True)
class RenderingResult:
    image_bytes: bytes
    xml_file: str


class RenderingEngine:
    def __init__(self,
                 number_rendering_servers: int,
                 server_init_timeout_seconds: int
                 ):
        with open(STANDARD_BIKE_RESOURCE, "r") as file:
            self.standard_bike_xml = file.read()
        self._rendering_client = self._init_rendering_client(number_rendering_servers, server_init_timeout_seconds)

    def render_xml(self, bike_xml: str) -> RenderingResult:
        return RenderingResult(image_bytes=(self._render(bike_xml)), xml_file=bike_xml)

    # def render_biked(self, biked: dict, rider_dims) -> RenderingResult:
    #     xml = FILE_BUILDER.build_cad_from_biked(biked, self.standard_bike_xml, rider_dims)
    #     return RenderingResult(image_bytes=(self._render(xml)), xml_file=xml)

    def render_clip(self, clip: dict, rider_dims=None) -> RenderingResult:
        xml = FILE_BUILDER.build_cad_from_clip(clip, self.standard_bike_xml, rider_dims)
        return RenderingResult(image_bytes=(self._render(xml)), xml_file=xml)

    def _render(self, xml: str) -> bytes:
        return self._rendering_client.render(xml)

    def _init_rendering_client(self,
                               number_rendering_servers: int,
                               timeout_seconds: int
                               ):
        if number_rendering_servers > 1:
            manager = MultiThreadedBikeCadServerManager(number_servers=number_rendering_servers,
                                                        timeout_seconds=timeout_seconds)
            return RenderingClient(server_manager=manager)
        else:
            return RenderingClient(server_manager=SingleThreadedBikeCadServerManager(timeout_seconds))
