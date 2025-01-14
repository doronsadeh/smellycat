import random
import geopy.distance
import matplotlib
from matplotlib import colors, cm

from app.geo_coloring import GeoColoring


def generate_random_hike(start_lat, start_lon, total_distance_km=1, step_distance_m=5):
    """
    Generates a random hike path.

    Parameters:
        start_lat (float): Starting latitude.
        start_lon (float): Starting longitude.
        total_distance_km (float): Total hike length in kilometers.
        step_distance_m (float): Step distance in meters for each segment.

    Returns:
        list: A list of [latitude, longitude] coordinates.
    """
    hike_path = [[start_lat, start_lon]]
    current_point = (start_lat, start_lon)
    total_distance_m = total_distance_km * 1000  # Convert to meters

    while total_distance_m > 0:
        # Generate a random direction (bearing in degrees)
        bearing = random.uniform(0, 270)

        # Compute the next point
        next_point = geopy.distance.distance(meters=step_distance_m).destination(current_point, bearing)

        # Update the path and remaining distance
        current_point = (next_point.latitude, next_point.longitude)
        hike_path.append([current_point[0], current_point[1]])
        total_distance_m -= step_distance_m

    return hike_path

if __name__ == "__main__":
    def generate_gradient_color(value, cmap_name="viridis"):
        """
        Maps a value (0-1) to a color in the given colormap.
        """
        norm = colors.Normalize(vmin=0, vmax=1)
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        color = cmap(norm(value))
        return f'rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, 0.8)'  # 80% transparency


    # Example: Generate a 1 km hike starting in Israel (Tel Aviv area)
    start_latitude = 32.461692
    start_longitude = 34.962014
    hike_coordinates = generate_random_hike(start_latitude, start_longitude, total_distance_km=15, step_distance_m=5)

    gcol = GeoColoring()

    geomap = None
    for c in hike_coordinates:
        geomap, _ = gcol.create_colored_hexagon_map(geomap, c[1], c[0], resolution=13, color=generate_gradient_color(random.random()))

    # Save the map to an HTML file
    geomap.save("geomap.html")

    print("Map saved to 'geomap.html'. Open it in a browser to view.")
