# Smelly Cat

![Smelly Cat Logo](./imgs/smellycat.png)

### Background

Years ago I happened to eat lunch at the SCO university open air auditorium. Some lecturer was supposed to come up
and give a fun insight about ... something. To my amazement Kurt Vonnegut took the stage, and addressed the audience
with the following question: "How do rhinoceroses see the world?"

The answer was: via **smell**.

> “No matter what is doing the creating, I have to say that the giraffe and the rhinoceros are ridiculous.”
> ― Kurt Vonnegut, Timequake

Well, that got my attention, 30 years ago. I recently remembered that for some odd reason
and decided to build on the late Vonnegut's vision. The system quickly came to be using
a Bosch multi-sensor (BME688), an Arduino (ESP32 Feather), and a Raspberry Pi (Zero WH)
connected to a GPS module (NEO v6).

The result of this effort was a raw smell graph, and some other nice sensor readings,
such as humidity, barometric pressure and temperature (which I currently do not use),
and a finalized realtime smell map.

![Smell Graphs](./imgs/smell-graphs.png)
<div style="text-align: center;font-size: small">Figure 1: Sensors Readouts</div>

To get the smell map in such a way that different smells get different colors while keeping
similar smells as similar colors, I had to employ a mapping from the 8-fold vector space
provided by the 8 gas sensors on the BME688, onto
a [metric color space](https://en.wikipedia.org/wiki/Oklab_color_space#:~:text=The%20Oklab%20color%20space%20is,stability%20and%20ease%20of%20implementation.).
The chosen metric color space was the [OkLAB](https://en.wikipedia.org/wiki/Oklab_color_space) color space.
Using PCA to transform the 8-fold sensors vector onto the lower dimensionality OkLAB colr space resulted
in a color map that was both accurate, and kept the smell-similarity.

Finally, the GPS location was cross-referenced with the sensor readout, to provide a location
where a smelled-colored hexagon was placed. Once done a median filter was applied over the hexagons
to flatten the colors into a smooth color field.

![Smell Map](./imgs/smell-map.png)
<div style="text-align: center;font-size: small">Figure 2: Geo Smell Dispersion Map</div>

### Architecture

TBD

### Usage

TBD

### Notes

- Here is
  a [GPS lib sample project](https://maker.pro/raspberry-pi/tutorial/how-to-use-a-gps-receiver-with-raspberry-pi-4).
- And another [tutorial](https://maker.pro/raspberry-pi/tutorial/how-to-read-gps-data-with-python-on-a-raspberry-pi). 