import cv2
import folium
import h3
import numpy as np
from PIL import Image
from selenium import webdriver
from matplotlib import cm, pyplot as plt
from matplotlib.colors import Normalize

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
        self.hexagons = {}

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

        self.hexagons[hex_id] = [float(x) for x in color.replace('rgba(', '').replace(')', '').split(',')]

        all_lats = []
        all_lons = []
        for hex_id in self.hexagons:
            hex_boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
            lats, lons = zip(*hex_boundary)
            all_lats.extend(lats)
            all_lons.extend(lons)

        # Create a Folium map centered on the given coordinates
        if geomap is None:
            geomap = folium.Map(location=[lon, lat], zoom_start=18)

        # Add the hexagon as a polygon to the map
        _polyhex = folium.Polygon(
            locations=hex_boundary,
            color=None,  # No border
            fill=True,
            fill_color=color,
            fill_opacity=0.35  # Semi-transparent fill
        )

        _polyhex.add_to(geomap)

        return geomap, {
            "min_lat": min(all_lats),
            "max_lat": max(all_lats),
            "min_lon": min(all_lons),
            "max_lon": max(all_lons)
        }

    def latlon_to_pixels(self, lat, lon, bounds, img_size):
        """
        Convert latitude and longitude to pixel coordinates.

        Parameters:
            lat (float): Latitude to convert.
            lon (float): Longitude to convert.
            bounds (dict): Map bounds with min/max lat/lon.
            img_size (tuple): Size of the map image in pixels (width, height).

        Returns:
            tuple: (x, y) pixel coordinates.
        """
        min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
        min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]
        width, height = img_size

        # Normalize latitude and longitude
        x = int((lon - min_lon) / (max_lon - min_lon) * width)
        y = int((max_lat - lat) / (max_lat - min_lat) * height)  # Invert Y-axis
        return x, y

    def capture_and_crop_bounding_box(self, geomap, html_file, bounding_box, output_image="cropped_map.jpg", img_size=(1024, 768)):
        """
        Capture the Folium map and crop it to the polygon bounding box.

        Parameters:
            geomap (folium.Map): Folium map object.
            html_file (str): Path to the HTML file of the map.
            bounding_box (dict): Bounding box with min/max lat/lon.
            output_image (str): Path to save the cropped image.
            img_size (tuple): Size of the browser window (width, height) in pixels.
        """
        # Initialize a Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument(f"--window-size={img_size[0]},{img_size[1]}")
        options.add_argument("--disable-color-correct-rendering")
        options.add_argument("--disable-gpu")

        driver = webdriver.Chrome(options=options)

        # Load the map HTML file
        driver.get(f"file://{html_file}")

        # Take a screenshot
        screenshot_path = "screenshot.png"
        driver.save_screenshot(screenshot_path)
        driver.quit()

        # Convert bounding box to pixel coordinates
        left, upper = self.latlon_to_pixels(bounding_box["max_lat"], bounding_box["min_lon"], bounding_box, img_size)
        right, lower = self.latlon_to_pixels(bounding_box["min_lat"], bounding_box["max_lon"], bounding_box, img_size)

        # Crop the screenshot to the bounding box
        with Image.open(screenshot_path) as img:
            cropped_img = img.crop((left, upper, right, lower))
            cropped_img.save(output_image, "PNG")
        print(f"Cropped image saved at '{output_image}'")

        # Load the uploaded image
        image = cv2.imread(output_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add alpha channel
        if image.shape[2] != 4:
            r, g, b = cv2.split(image)
            alpha = np.ones_like(b) * 255  # Fully opaque alpha channel
            image = cv2.merge((r, g, b, alpha))

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to detect polygons
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the polygons
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the center of the image
        image_center = (image.shape[1] // 2, image.shape[0] // 2)  # (x, y)

        # Find the contour closest to the center
        min_distance = float('inf')
        closest_contour = None

        for contour in contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M['m00'] != 0:  # Avoid division by zero
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                continue  # Skip contours with no area

            # Calculate the distance from the center of the image
            distance = np.sqrt((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2)

            # Check if this is the closest contour so far
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

        # Get the bounding rectangle of the closest contour
        x, y, w, h = cv2.boundingRect(closest_contour)

        # Crop the region within the contour
        cropped_image = image[y:y + h, x:x + w]

        # Define white color threshold
        threshold = 200  # Adjust as needed

        # Iterate through the image and set white pixels to transparent
        for y in range(cropped_image.shape[0]):
            for x in range(cropped_image.shape[1]):
                r, g, b, a = cropped_image[y, x]
                if b > threshold and g > threshold and r > threshold:
                    # Set alpha to 0 for transparency
                    cropped_image[y, x] = (r, g, b, 0)

        # Apply a median blur to the RGBA image (only on RGB channels, not transparency)
        blurred_image = cropped_image.copy()
        blurred_image[:, :, :3] = cv2.medianBlur(cropped_image[:, :, :3], ksize=17)

        # Save the blurred image
        blurred_image_path = "cropped_transparent_polygons_image.png"
        Image.fromarray(blurred_image).save(blurred_image_path)

        # # Display the blurred image
        # plt.imshow(blurred_image)
        # plt.title("Blurred Polygons with Transparency")
        # plt.axis("off")
        # plt.show()

        # Calculate the bounding box for non-transparent areas in the RGBA image
        alpha_channel = blurred_image[:, :, 3]  # Extract the alpha channel
        coords = cv2.findNonZero(alpha_channel)  # Find non-zero alpha pixels
        x, y, w, h = cv2.boundingRect(coords)  # Calculate the bounding box

        # Crop the image to the transparent bounding box
        cropped_to_transparent_bbox = blurred_image[y:y + h, x:x + w]

        # Define the border sizes
        top, bottom, left, right = 150, 150, 150, 150

        # Create the transparent border
        expanded_image = cv2.copyMakeBorder(
            cropped_to_transparent_bbox,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0, 0)  # Transparent border in BGRA
        )

        # Save and display the cropped image
        expanded_image_path = "blurred_transparent_polygons_image.png"
        Image.fromarray(expanded_image).save(expanded_image_path)

        # Display the cropped image
        # plt.imshow(cropped_to_transparent_bbox)
        # plt.title("Cropped to Transparent Bounding Box")
        # plt.axis("off")
        # plt.show()


        # Add the blurred image back to the map using ImageOverlay
        folium.raster_layers.ImageOverlay(
            name="Blurred Polygons",
            image="blurred_transparent_polygons_image.png",
            bounds=[[bounding_box['min_lat'], bounding_box['min_lon']],
                    [bounding_box['max_lat'], bounding_box['max_lon']]],  # Lat/Lon bounds
            opacity=1.0
        ).add_to(geomap)

        # Add layer control to toggle the overlay
        # folium.LayerControl().add_to(geomap)

        return geomap
