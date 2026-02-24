"""Parser for mower vector map data from the Dreame batch device data API.

The batch API returns map data split across numbered keys (MAP.0, MAP.1, ...).
These chunks must be concatenated in numeric order to form a complete JSON string.
The JSON contains polygon-based zone boundaries, navigation paths, and metadata.
"""
import json
import logging
import re
import time

from .types import (
    MowerMapBoundary,
    MowerMowPath,
    MowerPath,
    MowerVectorMap,
    MowerZone,
)

_LOGGER = logging.getLogger(__name__)

# Sentinel value in M_PATH data marking a path segment break
_PATH_SENTINEL = (32767, -32768)


def reassemble_map_chunks(batch_data: dict, prefix: str) -> str | None:
    """Reassemble chunked data from batch API into a single string.

    Keys like MAP.0, MAP.1, ... MAP.N are concatenated in numeric order.
    The MAP.info key is skipped (it's metadata, not map content).

    Args:
        batch_data: dict from get_batch_device_datas response
        prefix: key prefix to match, e.g. "MAP" or "M_PATH"

    Returns:
        Concatenated string, or None if no matching keys found.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)$")
    chunks = []
    for key, value in batch_data.items():
        match = pattern.match(key)
        if match:
            chunks.append((int(match.group(1)), value))

    if not chunks:
        return None

    chunks.sort(key=lambda x: x[0])
    return "".join(value for _, value in chunks)


def _parse_polygon_list(data_map: dict) -> list:
    """Parse a dataType:Map structure containing polygon entries."""
    if not data_map or data_map.get("dataType") != "Map":
        return []
    return data_map.get("value", [])


def _extract_path_coords(path_list: list) -> list[tuple[int, int]]:
    """Convert [{"x": ..., "y": ...}, ...] to [(x, y), ...]."""
    return [(p["x"], p["y"]) for p in path_list]


def parse_mower_map(map_json_str: str) -> MowerVectorMap:
    """Parse a single map JSON string into a MowerVectorMap.

    Args:
        map_json_str: JSON string for one map (after unescaping from the chunk array).

    Returns:
        MowerVectorMap with zones, paths, boundary, etc.
    """
    data = json.loads(map_json_str)
    vmap = MowerVectorMap()

    # Parse mowing area zones
    for entry in _parse_polygon_list(data.get("mowingAreas", {})):
        zone_id, zone_data = entry[0], entry[1]
        vmap.zones.append(MowerZone(
            zone_id=zone_id,
            path=_extract_path_coords(zone_data.get("path", [])),
            name=zone_data.get("name", ""),
            zone_type=zone_data.get("type", 0),
            shape_type=zone_data.get("shapeType", 0),
            area=zone_data.get("area", 0),
            time=zone_data.get("time", 0),
            etime=zone_data.get("etime", 0),
        ))

    # Parse forbidden areas
    for entry in _parse_polygon_list(data.get("forbiddenAreas", {})):
        zone_id, zone_data = entry[0], entry[1]
        vmap.forbidden_areas.append(MowerZone(
            zone_id=zone_id,
            path=_extract_path_coords(zone_data.get("path", [])),
            name=zone_data.get("name", ""),
            zone_type=zone_data.get("type", 0),
        ))

    # Parse navigation paths between zones
    for entry in _parse_polygon_list(data.get("paths", {})):
        path_id, path_data = entry[0], entry[1]
        vmap.paths.append(MowerPath(
            path_id=path_id,
            path=_extract_path_coords(path_data.get("path", [])),
            path_type=path_data.get("type", 0),
        ))

    # Parse boundary
    boundary = data.get("boundary")
    if boundary:
        vmap.boundary = MowerMapBoundary(
            x1=boundary["x1"], y1=boundary["y1"],
            x2=boundary["x2"], y2=boundary["y2"],
        )

    vmap.total_area = data.get("totalArea", 0)
    vmap.name = data.get("name", "")
    vmap.map_index = data.get("mapIndex", 0)
    vmap.last_updated = time.time()

    return vmap


def parse_mow_paths(batch_data: dict) -> list[MowerMowPath]:
    """Parse M_PATH.* keys from batch data into MowerMowPath objects.

    Each M_PATH.N key contains a coordinate array for zone N.
    Coordinates are comma-separated pairs: [x1,y1],[x2,y2],...
    The sentinel [32767,-32768] marks a segment break.

    Args:
        batch_data: dict from get_batch_device_datas response

    Returns:
        List of MowerMowPath objects.
    """
    pattern = re.compile(r"^M_PATH\.(\d+)$")
    paths = []

    for key, value in batch_data.items():
        match = pattern.match(key)
        if not match:
            continue

        zone_id = int(match.group(1))
        value = value.strip()

        if not value or value == "[]":
            continue

        # Parse the coordinate pairs from the string
        # Format: [x,y],[x,y],... or just x,y,x,y,...
        # We need to handle: ",[32767,-32768],[10,20],[30,40]"
        # Strip leading/trailing brackets and commas
        coords_str = value.strip("[], ")
        if not coords_str:
            continue

        # Parse all numbers
        numbers = re.findall(r"-?\d+", coords_str)
        if len(numbers) % 2 != 0:
            _LOGGER.warning("Odd number of coordinates in M_PATH.%d, skipping last", zone_id)
            numbers = numbers[:-1]

        # Build coordinate pairs, splitting on sentinel
        segments = []
        current_segment = []
        for i in range(0, len(numbers), 2):
            x, y = int(numbers[i]), int(numbers[i + 1])
            if (x, y) == _PATH_SENTINEL:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append((x, y))

        if current_segment:
            segments.append(current_segment)

        if segments:
            paths.append(MowerMowPath(zone_id=zone_id, segments=segments))

    return paths


def parse_batch_map_data(batch_data: dict) -> MowerVectorMap | None:
    """Parse complete batch device data response into a MowerVectorMap.

    This is the main entry point. It:
    1. Reassembles MAP.* chunks into a full JSON string
    2. Parses the JSON array (may contain multiple maps by mapIndex)
    3. Returns the primary map (mapIndex 0)
    4. Attaches mowing paths from M_PATH.* keys

    Args:
        batch_data: Full response dict from get_batch_device_datas

    Returns:
        MowerVectorMap for the primary map, or None if parsing fails.
    """
    if not batch_data:
        return None

    raw_map = reassemble_map_chunks(batch_data, "MAP")
    if not raw_map:
        _LOGGER.debug("No MAP chunks found in batch data")
        return None

    try:
        # The reassembled string is a JSON array of JSON strings
        map_array = json.loads(raw_map)
    except json.JSONDecodeError:
        _LOGGER.warning("Failed to parse reassembled MAP data as JSON")
        return None

    if not map_array:
        return None

    # Parse each map entry — they're JSON strings within the array
    primary_map = None
    for map_json_str in map_array:
        try:
            vmap = parse_mower_map(map_json_str)
            if vmap.map_index == 0:
                primary_map = vmap
        except (json.JSONDecodeError, KeyError, TypeError) as ex:
            _LOGGER.warning("Failed to parse map entry: %s", ex)
            continue

    if primary_map is None and map_array:
        # Fall back to first entry if no mapIndex 0 found
        try:
            primary_map = parse_mower_map(map_array[0])
        except (json.JSONDecodeError, KeyError, TypeError) as ex:
            _LOGGER.warning("Failed to parse fallback map entry: %s", ex)
            return None

    if primary_map is None:
        return None

    # Attach mowing paths
    primary_map.mow_paths = parse_mow_paths(batch_data)

    return primary_map
