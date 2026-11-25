"""current_date and weather tools."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx

from forgeai.tools.base import ToolContext, ToolError

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"


class CurrentDate:
    name = "current_date"

    def __init__(self, context: ToolContext) -> None:
        self._context = context

    def run(self, args: dict[str, Any]) -> str:
        today = self._context.today()
        return f"{today.isoformat()} ({today.strftime('%A')})"


class Weather:
    name = "weather"

    def __init__(self, context: ToolContext) -> None:
        self._context = context

    def run(self, args: dict[str, Any]) -> str:
        try:
            lat, lon = float(args["lat"]), float(args["lon"])
            day = date.fromisoformat(str(args.get("date") or self._context.today().isoformat()))
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError("weather needs numeric 'lat', 'lon' and an ISO 'date' (YYYY-MM-DD)") from exc
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ToolError("lat/lon out of range")
        today = self._context.today()
        url = ARCHIVE_URL if day < today - timedelta(days=5) else FORECAST_URL
        params = {
            "latitude": lat, "longitude": lon, "daily": _DAILY,
            "start_date": day.isoformat(), "end_date": day.isoformat(), "timezone": "auto",
        }
        try:
            response = self._context.http.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"weather service unavailable: {exc}") from exc
        daily = response.json().get("daily", {})
        if not daily.get("time"):
            raise ToolError("no weather data for that date and location")
        return (
            f"{day.isoformat()} at ({lat}, {lon}): max {daily['temperature_2m_max'][0]} C, "
            f"min {daily['temperature_2m_min'][0]} C, precipitation {daily['precipitation_sum'][0]} mm, "
            f"max wind {daily['wind_speed_10m_max'][0]} km/h"
        )
