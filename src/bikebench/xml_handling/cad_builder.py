import pandas as pd
import numpy as np
from typing import Optional, Sequence

from bikebench.transformation.one_hot_encoding import decode_to_mixed
from bikebench.xml_handling.bike_xml_handler import BikeXmlHandler
from bikebench.xml_handling.bikebench_to_bcad import bikebench_to_cad

OPTIMIZED_TO_CAD = {
    "ST Angle": "Seat angle",
    "HT Length": "Head tube length textfield",
    "HT Angle": "Head angle",
    "HT LX": "Head tube lower extension2",
    'Stack': 'Stack',
    "ST Length": "Seat tube length",
    "Seatpost LENGTH": "Seatpost LENGTH",
    "Saddle height": "Saddle height",
    "Stem length": "Stem length",
    "Crank length": "Crank length",
    "Headset spacers": "Headset spacers",
    "Stem angle": "Stem angle",
    "Handlebar style": "Handlebar style",
}


class BikeCadFileBuilder:
    # def build_cad_from_biked(self, biked: dict, seed_bike_xml: str, rider_dims = None) -> str:
    #     xml_handler = BikeXmlHandler()
    #     xml_handler.set_xml(seed_bike_xml)
    #     for response_key, cad_key in OPTIMIZED_TO_CAD.items():
    #         self._update_xml(xml_handler, cad_key, biked[response_key])
    #     if rider_dims:
    #         xml_handler.add_or_update("Display RIDER", "true")
    #     return xml_handler.get_content_string()

    def build_cad_from_clip(self, clip: dict, seed_bike_xml: str, rider_dims: Optional[Sequence[float]] = None) -> str:
        xml_handler = BikeXmlHandler()
        xml_handler.set_xml(seed_bike_xml)
        target_dict = self._to_cad_dict(clip)
        self._update_values(xml_handler, target_dict)
        if rider_dims is not None: #upper_leg, lower_leg, arm_length, torso_length, neck_and_head_length, torso_width
            rider_dims = 1000 * np.array(rider_dims).flatten()  # Convert to mm
            xml_handler.add_or_update("Display RIDER", "true")
            xml_handler.add_or_update("UPPERARMLENGTH", str(rider_dims[2]/2))
            xml_handler.add_or_update("LOWERARMLENGTH", str(rider_dims[2]/2))
            xml_handler.add_or_update("TORSOLENGTH", str(rider_dims[3]))
            xml_handler.add_or_update("LOWERLEGLENGTH", str(rider_dims[1]))
            xml_handler.add_or_update("UPPERLEGLENGTH", str(rider_dims[0]))
            xml_handler.add_or_update("SHOULDERroll", "0")
            handpos = target_dict["Handlebar style"]
            handlookup = {"0": "3", "1": "1", "2": "2"} #0,1,2 -> 3,1,2 (bikecad's 0 is aerobars)
            xml_handler.add_or_update("Hand position", handlookup[handpos])

        return xml_handler.get_content_string()

    def _to_cad_dict(self, bike: dict):
        bike_complete = bikebench_to_cad(pd.DataFrame.from_records([bike])).iloc[0]
        decoded_values = decode_to_mixed(pd.DataFrame.from_records([bike_complete]))
        decoded_values = decoded_values.iloc[0].to_dict()
        return decoded_values

    def _update_xml(self, xml_handler, cad_key, desired_value):
        entry = xml_handler.find_entry_by_key(cad_key)
        if entry:
            xml_handler.update_entry_value(entry, str(desired_value))
        else:
            xml_handler.add_new_entry(cad_key, str(desired_value))

    def _update_values(self, handler, bike_dict):
        num_updated = 0
        for k, v in bike_dict.items():
            parsed = self._parse(v)
            if parsed is not None:
                num_updated += 1
                self._update_value(parsed, handler, k)

    def _parse(self, v):
        handled = self._handle_numeric(v)
        handled = self._handle_bool(str(handled))
        return handled

    def _update_value(self, handled, xml_handler, k):
        xml_handler.update_if_exists(k, handled)

    def _handle_numeric(self, v):
        if str(v).lower() == 'nan':
            return None
        if type(v) in [int, float]:
            v = int(v)
        return v

    def _handle_bool(self, param):
        if param.lower().title() in ['True', 'False']:
            return param.lower()
        return param
