from datetime import date
from pathlib import Path

import httpx
import pytest

from forgeai.tools import REGISTRY, ToolContext, ToolError, build_tools, tool_description
from forgeai.tools.calculator import Calculator, evaluate, format_number

SMOKE_DOCS = Path(__file__).parents[1] / "data" / "_smoke" / "docs"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", "7"),
        ("(10 - 4) / 4", "1.5"),
        ("round(120 * 14.5 * (1 - 0.08), 2)", "1600.8"),
        ("-3 ** 2", "-9"),
        ("max(2, 9, 4) // 2", "4"),
        ("100 % 7", "2"),
        ("sqrt(144)", "12"),
        ("0.1 + 0.2", "0.3"),
    ],
)
def test_calculator_values(expression: str, expected: str) -> None:
    assert Calculator().run({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "[1,2][0]",
        "(lambda: 1)()",
        "x + 1",
        "2 ** 1000",
        "1 / 0",
        "'a' * 3",
        "().__class__",
        "1 +",
        "",
    ],
)
def test_calculator_rejects_unsafe_or_invalid(expression: str) -> None:
    with pytest.raises(ToolError):
        Calculator().run({"expression": expression})


def test_calculator_never_uses_eval() -> None:
    source = (Path(__file__).parents[1] / "forgeai" / "tools" / "calculator.py").read_text(encoding="utf-8")
    assert "eval(" not in source.replace("_eval(", "").replace("evaluate(", "")


def test_format_number() -> None:
    assert format_number(3.0) == "3" and format_number(2.5) == "2.5"


def test_csv_lookup_filters_and_columns() -> None:
    tools = build_tools(["csv_lookup"], ToolContext(corpus_dir=SMOKE_DOCS))
    result = tools["csv_lookup"].run(
        {"file": "rooms.csv", "filters": {"projector": "yes", "hourly_fee_usd": {"op": "<", "value": 30}},
         "columns": ["room", "capacity"]}
    )
    assert result == "1 matching rows\nroom: Studio A | capacity: 12"


def test_csv_lookup_contains_and_no_match() -> None:
    tool = build_tools(["csv_lookup"], ToolContext(corpus_dir=SMOKE_DOCS))["csv_lookup"]
    assert "Studio B" in tool.run({"file": "rooms.csv", "filters": {"room": {"op": "contains", "value": "studio"}}})
    assert tool.run({"file": "rooms.csv", "filters": {"room": "Nowhere"}}) == "0 matching rows"


@pytest.mark.parametrize(
    "args",
    [
        {"file": "../data/_smoke/agent.yaml"},
        {"file": "missing.csv"},
        {"file": "rooms.csv", "filters": {"nope": 1}},
        {"file": "rooms.csv", "columns": ["nope"]},
        {"file": "rooms.csv", "filters": {"capacity": {"op": ">", "value": "lots"}}},
        {},
    ],
)
def test_csv_lookup_errors_are_toolerrors(args: dict) -> None:
    tool = build_tools(["csv_lookup"], ToolContext(corpus_dir=SMOKE_DOCS))["csv_lookup"]
    with pytest.raises(ToolError):
        tool.run(args)


def test_current_date_uses_injected_clock() -> None:
    ctx = ToolContext(today=lambda: date(2026, 3, 14))
    assert build_tools(["current_date"], ctx)["current_date"].run({}) == "2026-03-14 (Saturday)"


def _weather_context(handler) -> ToolContext:
    return ToolContext(today=lambda: date(2026, 3, 14), http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_weather_parses_open_meteo_response() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"daily": {
            "time": ["2026-03-15"], "temperature_2m_max": [31.0], "temperature_2m_min": [26.5],
            "precipitation_sum": [2.4], "wind_speed_10m_max": [18.0]}})

    result = build_tools(["weather"], _weather_context(handler))["weather"].run(
        {"lat": 1.29, "lon": 103.85, "date": "2026-03-15"})
    assert "max 31.0 C" in result and "precipitation 2.4 mm" in result
    assert "api.open-meteo.com/v1/forecast" in seen["url"]


def test_weather_uses_archive_for_old_dates_and_reports_failures() -> None:
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(503)

    tool = build_tools(["weather"], _weather_context(handler))["weather"]
    with pytest.raises(ToolError, match="unavailable"):
        tool.run({"lat": 1, "lon": 2, "date": "2026-01-01"})
    assert "archive-api" in urls[0]
    with pytest.raises(ToolError):
        tool.run({"lat": 999, "lon": 0, "date": "2026-03-15"})


def test_registry_has_exactly_four_tools_with_descriptions() -> None:
    assert set(REGISTRY) == {"csv_lookup", "calculator", "current_date", "weather"}
    assert {s.capability for s in REGISTRY.values()} == {
        "structured_lookup", "arithmetic", "temporal", "live_weather"}
    for name in REGISTRY:
        assert name in tool_description(name)


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ToolError):
        build_tools(["teleport"], ToolContext())
