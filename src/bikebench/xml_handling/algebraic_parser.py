from typing import Union


class AlgebraicParser:
    """Parses booleans and floats. Represents booleans as the floating point values 0 and 1."""

    def attempt_parse(self, value) -> Union[float, str]:
        """Attempts to parse a string value (converts the value to string defensively). Returns
        the string value stripped in case it fails to parse it. None values map to an empty string.
        """
        if value is None:
            return ""
        return self._parse_value(str(value).strip())

    def _parse_value(self, value: str) -> Union[float, str]:
        if self._is_bool(value.lower()):
            return float(value.lower() == "true")
        if self._is_float(value):
            return float(value)
        return value.strip()

    def _is_float(self, value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _is_bool(self, value: str) -> bool:
        return value in ["true", "false"]
