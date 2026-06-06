"""API data models for the LangGraph trip planner."""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class TripRequest(BaseModel):
    """Travel planning request."""

    city: str = Field(..., description="目的地城市", examples=["北京"])
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD", examples=["2025-06-01"])
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD", examples=["2025-06-03"])
    travel_days: int = Field(..., description="旅行天数", ge=1, le=30, examples=[3])
    transportation: str = Field(..., description="交通方式", examples=["公共交通"])
    accommodation: str = Field(..., description="住宿偏好", examples=["经济型酒店"])
    preferences: List[str] = Field(default_factory=list, description="旅行偏好标签")
    free_text_input: Optional[str] = Field(default="", description="额外要求")

    model_config = {
        "json_schema_extra": {
            "example": {
                "city": "北京",
                "start_date": "2025-06-01",
                "end_date": "2025-06-03",
                "travel_days": 3,
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "preferences": ["历史文化", "美食"],
                "free_text_input": "希望多安排一些博物馆",
            }
        }
    }


class Location(BaseModel):
    """Geographic location."""

    longitude: float = Field(..., description="经度")
    latitude: float = Field(..., description="纬度")


class Attraction(BaseModel):
    """Attraction information."""

    name: str
    address: str
    location: Location
    visit_duration: int
    description: str
    category: Optional[str] = "景点"
    rating: Optional[float] = None
    photos: Optional[List[str]] = Field(default_factory=list)
    poi_id: Optional[str] = ""
    image_url: Optional[str] = None
    ticket_price: int = 0


class Meal(BaseModel):
    """Meal recommendation."""

    type: str
    name: str
    address: Optional[str] = None
    location: Optional[Location] = None
    description: Optional[str] = None
    estimated_cost: int = 0


class Hotel(BaseModel):
    """Hotel recommendation."""

    name: str
    address: str = ""
    location: Optional[Location] = None
    price_range: str = ""
    rating: str = ""
    distance: str = ""
    type: str = ""
    estimated_cost: int = 0


class DayPlan(BaseModel):
    """Single-day itinerary."""

    date: str
    day_index: int
    description: str
    transportation: str
    accommodation: str
    hotel: Optional[Hotel] = None
    attractions: List[Attraction] = Field(default_factory=list)
    meals: List[Meal] = Field(default_factory=list)


class WeatherInfo(BaseModel):
    """Weather information for one day."""

    date: str
    day_weather: str = ""
    night_weather: str = ""
    day_temp: Union[int, str] = 0
    night_temp: Union[int, str] = 0
    wind_direction: str = ""
    wind_power: str = ""

    @field_validator("day_temp", "night_temp", mode="before")
    @classmethod
    def parse_temperature(cls, value):
        if isinstance(value, str):
            cleaned = value.replace("°C", "").replace("℃", "").replace("°", "").strip()
            try:
                return int(cleaned)
            except ValueError:
                return 0
        return value


class Budget(BaseModel):
    """Trip budget summary."""

    total_attractions: int = 0
    total_hotels: int = 0
    total_meals: int = 0
    total_transportation: int = 0
    total: int = 0


class TripPlan(BaseModel):
    """Complete travel plan."""

    city: str
    start_date: str
    end_date: str
    days: List[DayPlan]
    weather_info: List[WeatherInfo] = Field(default_factory=list)
    overall_suggestions: str
    budget: Optional[Budget] = None


class TripPlanResponse(BaseModel):
    """Trip planning API response."""

    success: bool
    message: str = ""
    data: Optional[TripPlan] = None
    elapsed_ms: Optional[int] = Field(default=None, description="服务端生成行程耗时，单位毫秒")


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = False
    message: str
    error_code: Optional[str] = None
