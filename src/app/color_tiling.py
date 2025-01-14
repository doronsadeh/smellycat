import folium
import h3
import numpy as np
from matplotlib import cm, colors


# Helper function to generate a gradient color
def generate_gradient_color(value, cmap_name="viridis"):
    """
    Maps a value (0-1) to a color in the given colormap.
    """
    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap(cmap_name)
    color = cmap(norm(value))
    return f'rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, 0.8)'  # 80% transparency


# Create the base map
center_coords = [2.8220600027806313, 19.12493353251449]  # Example: San Francisco, change to your location
mymap = folium.Map(location=center_coords, zoom_start=7)

# Define a valid bounding polygon in GeoJSON format (must be closed and in [lon, lat] order)
bounding_polygon = {
    "type": "Polygon",
    "coordinates": [
        [
            [
                19.12493353251449,
                2.8220600027806313
            ],
            [
                19.12493353251449,
                1.0712491584803558
            ],
            [
                20.8572253966162,
                1.0712491584803558
            ],
            [
                20.8572253966162,
                2.8220600027806313
            ],
            [
                19.12493353251449,
                2.8220600027806313
            ]
        ]
    ]
}

# Generate hexagons using h3.polyfill with resolution 7
resolution = 7
hexagons = h3.polyfill(bounding_polygon, resolution)

# Check if the list is empty
if not hexagons:
    raise ValueError("No hexagons were generated. Check the bounding polygon and resolution.")

# Iterate over each hexagon and add it to the map
for hex_id in hexagons:
    # Get the boundary of the hexagon
    hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)

    # Assign a random value for coloring
    color_value = np.random.rand()

    # Get a color from the gradient
    color = generate_gradient_color(color_value)

    # Add the hexagon to the map
    folium.Polygon(
        locations=hex_boundary,
        color=None,  # No border
        fill=True,
        fill_color=color,
        fill_opacity=0.5,  # 80% transparency
    ).add_to(mymap)

# Save the map to an HTML file
mymap.save("colored_hexagon_map_5km.html")

print("Map saved to 'colored_hexagon_map_5km.html'. Open it in a browser to view.")
