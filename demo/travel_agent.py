from datetime import datetime, timedelta

import requests

from fence.agents.agent import Agent
from fence.models.openai import GPT4omini
from fence.tools.base import BaseTool
from fence.utils.logger import setup_logging

logger = setup_logging(__name__, log_level="kill")  # Only show agentic messaging


class WeatherTool(BaseTool):
    """
    Tool to get weather forecast for a location.
    """

    def execute_tool(self, location: str, days: int = 7, **kwargs):
        """
        Execute the tool to get weather forecast for a location.

        :param location: Location name.
        :param days: Number of days to forecast.
        :return: String containing weather forecast summary.
        """

        # Get coordinates for the location
        latitude, longitude = self.get_coordinates(location=location)
        if latitude is None or longitude is None:
            error_msg = f"Error: Could not get coordinates for location: {location}"
            logger.error(error_msg)
            return error_msg

        # Get weather forecast
        try:
            weather_forecast = self.get_weather_forecast(
                latitude=latitude, longitude=longitude, days=days
            )
            response = (
                f"Here is the weather forecast for {location}:\n\n{weather_forecast}"
            )
            return response

        except Exception as e:
            error_msg = f"Error getting weather forecast: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_coordinates(self, location: str):
        """
        Get the coordinates of a location.

        :param location: Location name.
        :return: Latitude and longitude of the location.
        """
        # Nominatim API endpoint for geocoding
        geocode_url = "http://nominatim.openstreetmap.org/search"
        params = {"q": location, "format": "json", "limit": 1}  # Get the first result

        # Headers to avoid 403 error
        headers = {"User-Agent": "Fence - agent demo"}

        # Send the GET request to Nominatim
        response = requests.get(geocode_url, params=params, headers=headers)
        if response.status_code == 200 and response.json():
            location_data = response.json()[0]
            return float(location_data["lat"]), float(location_data["lon"])
        else:
            logger.error(
                f"Could not get coordinates for the location: {location} - Status code: {response.status_code}, Response: {response.text}"
            )
            return None, None

    def get_weather_forecast(self, latitude: float, longitude: float, days: int):
        """
        Get the weather forecast for a location.

        :param latitude: Latitude of the location.
        :param longitude: Longitude of the location.
        :param days: Number of days to forecast.
        :return: Weather forecast data with min/max temperatures and their dates, plus average temperature.
        """
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&start_date={start_date}&end_date={end_date}"
            "&timezone=auto"
        )

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            daily_data = data.get("daily", {})

            # Extract the lists of dates and temperatures
            dates = daily_data.get("time", [])
            max_temps = daily_data.get("temperature_2m_max", [])
            min_temps = daily_data.get("temperature_2m_min", [])
            precipitation = daily_data.get("precipitation_sum", [])

            # Find the maximum and minimum temperatures and their dates
            max_temp = max(max_temps)
            min_temp = min(min_temps)
            max_temp_date = dates[max_temps.index(max_temp)]
            min_temp_date = dates[min_temps.index(min_temp)]

            # Calculate average temperature
            avg_temp = sum(
                [(max_t + min_t) / 2 for max_t, min_t in zip(max_temps, min_temps)]
            ) / len(max_temps)

            # Create the weather summary string
            weather_forecast = (
                f"Highest temperature will be {max_temp}°C on {max_temp_date}\n"
                f"Lowest temperature will be {min_temp}°C on {min_temp_date}\n"
                f"Average temperature across all days: {avg_temp:.1f}°C\n\n"
                f"Daily Breakdown:\n"
            )

            # Add daily details
            for date, max_t, min_t, precip in zip(
                dates, max_temps, min_temps, precipitation
            ):
                weather_forecast += (
                    f"{date}: Max Temp: {max_t}°C, Min Temp: {min_t}°C, "
                    f"Precipitation: {precip}mm\n"
                )

            return weather_forecast
        else:
            error_msg = f"Error: Could not get weather forecast. Status code: {response.status_code}"
            logger.error(error_msg)
            return error_msg


if __name__ == "__main__":

    # Setup logging
    setup_logging(log_level="info", are_you_serious=False)

    # Create a weather agent
    weather_agent = Agent(
        identifier="weather_agent",
        model=GPT4omini(),
        description="You are a very helpful and friendly weather man. You have a weather forecast tool that helps you get more information about specific locations. Always respond to the user. Ask more information if necessary.",
        tools=[WeatherTool()],
    )

    # Create a travel agent
    travel_agent = Agent(
        model=GPT4omini(),
        description="You are a travel agent. You are jovial and cheerful, and help people with information about possible travel destinations.",
        delegates=[weather_agent],
    )

    # Run the agent
    while True:
        prompt = input("😎: ")
        response = travel_agent.run(prompt)
        if response == "quit":
            break
        print(f"🤺️: {response}")
