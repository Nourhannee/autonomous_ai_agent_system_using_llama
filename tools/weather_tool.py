"""
Tool: Weather
Fetches current weather for any city using Open-Meteo (free, no API key).
"""

import urllib.request
import urllib.parse
import json
from langchain_core.tools import tool


@tool
def weather_tool(city: str) -> str:
    """
    Get current weather for any city in the world.
    Input: city name e.g. Cairo, London, New York.
    Returns temperature, humidity, wind speed, and conditions.
    """
    city = city.strip().strip('"').strip("'")
    if not city:
        return "Error: Please provide a city name."
    try:
        # Geocode
        url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        results = data.get("results")
        if not results:
            return f"Error: City not found: '{city}'"
        r = results[0]
        lat, lon = r["latitude"], r["longitude"]
        display_name = f"{r['name']}, {r.get('country', '')}"

        # Weather
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "weather_code,apparent_temperature,precipitation"
            "&temperature_unit=celsius&wind_speed_unit=kmh"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())

        c = data["current"]
        wmo = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 80: "Showers", 95: "Thunderstorm",
        }
        condition = wmo.get(c.get("weather_code", 0), "Unknown")
        return (
            f"Weather in {display_name}:\n"
            f"  Condition:     {condition}\n"
            f"  Temperature:   {c['temperature_2m']}C (feels like {c['apparent_temperature']}C)\n"
            f"  Humidity:      {c['relative_humidity_2m']}%\n"
            f"  Wind speed:    {c['wind_speed_10m']} km/h\n"
            f"  Precipitation: {c['precipitation']} mm\n"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"