import folium
import h3

"""
Resolution  Average Hexagon Edge Length (m)	Average Hexagon Area (km²)	Average Hexagon Area (m²)
    0	                    1107.7	                4,250	                    4,250,000
    1	                    418.7	                607.0	                    607,000
    2	                    158.2	                86.8	                    86,800
    3	                    59.8	                12.4	                    12,400
    4	                    22.6	                1.77	                    1,770
    5	                    8.6	                    0.254	                    254
    6	                    3.2	                    0.036	                    36
    7	                    1.2	                    0.005	                    5
    8	                    0.46	                0.0007	                    700
    9	                    0.17	                0.0001	                    100
    10	                    0.063	                0.00002	                    20
    11	                    0.023	                0.000003	                3
    12	                    0.0084	                0.0000005	                0.5
    13	                    0.0031	                0.0000001	                0.1
    14	                    0.0011	                0.00000001	                0.01
    15	                    0.0004	                0.000000001	                0.001

"""


class GeoColoring:
    def __init__(self):
        pass

    def get_hexagon_coordinates(self, lat, lon, resolution=15):
        """
        Compute the hexagon coordinates around a given latitude and longitude.

        Parameters:
            lat (float): Latitude of the central point.
            lon (float): Longitude of the central point.
            resolution (int): H3 resolution (default is 15 for ~1 square meter).

        Returns:
            list: List of [lat, lon] pairs representing the hexagon's boundary.
        """
        # Get the H3 index for the given lat/lon at the specified resolution
        hex_id = h3.geo_to_h3(lat, lon, resolution)

        # Get the boundary of the hexagon
        hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)

        return hex_boundary, hex_id

    def create_colored_hexagon_map(self, geomap, lat, lon, resolution=7, color="blue"):
        """
        Create a Folium map with a colored hexagon around the given lat/lon.

        Parameters:
            lat (float): Latitude of the central point.
            lon (float): Longitude of the central point.
            resolution (int): H3 resolution for the hexagon.
            color (str): Fill color for the hexagon.

        Returns:
            folium.Map: Folium map object with the colored hexagon.
        """
        # Get hexagon coordinates
        hex_boundary, hex_id = self.get_hexagon_coordinates(lat, lon, resolution)

        # Create a Folium map centered on the given coordinates
        if geomap is None:
            geomap = folium.Map(location=[lon, lat], zoom_start=14)

        # Add the hexagon as a polygon to the map
        folium.Polygon(
            locations=hex_boundary,
            color=None,  # No border
            fill=True,
            fill_color=color,
            fill_opacity=0.75  # Semi-transparent fill
        ).add_to(geomap)

        return geomap
