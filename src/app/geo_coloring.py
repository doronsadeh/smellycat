import cv2
import folium
import h3
import numpy as np
from PIL import Image
from selenium import webdriver
from matplotlib import cm
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

    def capture_and_crop_bounding_box(self, geomap, html_file, bounding_box, output_image="cropped_map.jpg", img_size=(640, 480)):
        """
        Capture the Folium map and crop it to the polygon bounding box.

        Parameters:
            html_file (str): Path to the HTML file of the map.
            bounding_box (dict): Bounding box with min/max lat/lon.
            output_image (str): Path to save the cropped image.
            img_size (tuple): Size of the browser window (width, height) in pixels.
        """
        # Initialize a Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument(f"--window-size={img_size[0]},{img_size[1]}")
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
            cropped_img.save(output_image, "JPEG")
        print(f"Cropped image saved at '{output_image}'")

        # Load the uploaded image
        image = cv2.imread(output_image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

        # Display the original image
        # plt.imshow(image_rgb)
        # plt.title("Original Image")
        # plt.axis("off")
        # plt.show()

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to detect polygons
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the polygons
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detected contours
        contour_image = image_rgb.copy()
        cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)

        # Display the contours
        # plt.imshow(contour_image)
        # plt.title("Contours Detected")
        # plt.axis("off")
        # plt.show()

        # Calculate the areas of the polygons
        polygon_areas = [cv2.contourArea(cnt) for cnt in contours]

        # Normalize the colormap range to identify the polygons of the 'coolwarm' colormap range
        cmap = cm.get_cmap('coolwarm')
        norm = Normalize(vmin=0, vmax=255)

        # Create a mask for polygons within the coolwarm range
        coolwarm_mask = np.zeros_like(gray, dtype=np.uint8)

        # Iterate over contours and filter polygons within the colormap's range
        for cnt in contours:
            # Create a temporary mask for the current contour
            mask_temp = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(mask_temp, [cnt], -1, 255, thickness=cv2.FILLED)

            # Calculate the average color within the contour
            mean_color = cv2.mean(image_rgb, mask=mask_temp)[:3]  # Only RGB channels

            # Normalize the color values to [0, 1] for comparison with the colormap
            normalized_color = tuple(val / 255 for val in mean_color)
            colormap_color = cmap(norm(np.mean(normalized_color)))

            # Check if the alpha channel of the colormap output is non-zero
            if colormap_color[-1] > 0:  # Alpha > 0 means it falls in the colormap range
                cv2.drawContours(coolwarm_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Apply the new mask to isolate polygons in the coolwarm range
        masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=coolwarm_mask)

        # Find the bounding box for these polygons
        x_cw, y_cw, w_cw, h_cw = cv2.boundingRect(np.vstack(contours))

        # Crop the area containing the filtered polygons
        cropped_coolwarm_polygons_image = masked_image[y_cw:y_cw + h_cw, x_cw:x_cw + w_cw]

        # Save the cropped image
        cropped_coolwarm_polygons_image_path = "cropped_transparent_polygons_image.png"
        Image.fromarray(cropped_coolwarm_polygons_image).save(cropped_coolwarm_polygons_image_path)

        # Display the cropped image
        # plt.imshow(cropped_coolwarm_polygons_image)
        # plt.title("Cropped Coolwarm Polygons")
        # plt.axis("off")
        # plt.show()

        # Convert the masked image to RGBA format
        masked_image_rgba = cv2.cvtColor(masked_image, cv2.COLOR_RGB2RGBA)

        # Set the non-polygon area (black pixels in the mask) to transparent
        masked_image_rgba[gray == 0, 3] = 0  # Set alpha channel to 0 for non-polygon areas

        # Crop the area containing the polygons
        cropped_transparent_polygons_image = masked_image_rgba[y_cw:y_cw + h_cw, x_cw:x_cw + w_cw]

        # Save the cropped image with transparency
        cropped_transparent_polygons_image_path = "cropped_transparent_polygons_image.png"
        Image.fromarray(cropped_transparent_polygons_image).save(cropped_transparent_polygons_image_path)

        # Display the cropped image with transparency
        # plt.imshow(cropped_transparent_polygons_image)
        # plt.title("Cropped Polygons with Transparency")
        # plt.axis("off")
        # plt.show()

        # Apply a median blur to the RGBA image (only on RGB channels, not transparency)
        blurred_image = cropped_transparent_polygons_image.copy()
        blurred_image[:, :, :3] = cv2.medianBlur(cropped_transparent_polygons_image[:, :, :3], ksize=17)

        # Save the blurred image
        blurred_image_path = "blurred_transparent_polygons_image.png"
        Image.fromarray(blurred_image).save(blurred_image_path)

        # Display the blurred image
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

        # Save and display the cropped image
        cropped_to_transparent_bbox_path = "blurred_transparent_polygons_image.png"
        Image.fromarray(cropped_to_transparent_bbox).save(cropped_to_transparent_bbox_path)

        # Display the cropped image
        # plt.imshow(cropped_to_transparent_bbox)
        # plt.title("Cropped to Transparent Bounding Box")
        # plt.axis("off")
        # plt.show()
        #
        # Add the blurred image back to the map using ImageOverlay
        # folium.raster_layers.ImageOverlay(
        #     name="Blurred Polygons",
        #     image="blurred_transparent_polygons_image.png",
        #     bounds=[[bounding_box['min_lat'], bounding_box['min_lon']],
        #             [bounding_box['max_lat'], bounding_box['max_lon']]],  # Lat/Lon bounds
        #     opacity=1.0
        # ).add_to(geomap)

        # Add layer control to toggle the overlay
        # folium.LayerControl().add_to(geomap)

        return geomap
