"""LangGraph multi-agent trip planner.

This mirrors the original four-step learning flow:
1. attraction specialist
2. weather specialist
3. hotel specialist
4. planner specialist

The first three specialists run in parallel and each uses an LLM agent with
MCP tools, so the model still decides whether and how to call tools.
"""

from __future__ import annotations

import asyncio
import operator
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..models.schemas import Attraction, DayPlan, Location, Meal, TripPlan, TripRequest
from ..services.llm_service import get_llm
from ..services.mcp_service import filter_tools, get_amap_tools


ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

你可以自主决定是否调用工具；如果需要真实景点、地址、经纬度、POI信息，应优先调用高德地图工具。
请输出简洁但信息密度高的中文结果，保留工具返回中的关键名称、地址、坐标、类别和可用原始片段。
不要编造工具没有支持的 POI 细节。"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

你可以自主决定是否调用工具；如果需要实时或城市天气，应优先调用高德地图天气工具。
请输出可供后续行程规划使用的中文结果，包含天气、温度、风向、风力等信息。"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市、住宿偏好和景点上下文推荐合适酒店。

你可以自主决定是否调用工具；如果需要真实酒店名称、地址、坐标，应优先调用高德地图文本搜索工具。
请输出若干候选酒店及推荐理由，尽量保留地址、坐标、评分、价格区间等线索。"""

PLANNER_SYSTEM_PROMPT = """你是行程规划专家。请根据用户需求、景点信息、天气信息和酒店信息生成旅行计划。

要求:
1. 输出必须满足 TripPlan 结构。
2. weather_info 数组应尽量包含每一天的天气信息。
3. 每天安排 2-3 个景点，并考虑景点之间距离和游览时间。
4. 每天必须包含早餐、午餐、晚餐。
5. 每天推荐一个具体酒店；如果工具结果不足，可以明确标注为基于偏好的建议。
6. 景点经纬度优先使用工具结果；若缺失，给出合理占位并在 overall_suggestions 中说明。
7. 必须包含预算信息，包括门票、酒店、餐饮、交通和总费用。"""


class TripPlannerState(TypedDict, total=False):
    """State shared by LangGraph nodes."""

    request: TripRequest
    attraction_results: str
    weather_results: str
    hotel_results: str
    final_plan: TripPlan
    errors: Annotated[List[str], operator.add]


def _message_text(result: Any) -> str:
    """Extract text from a LangChain/LangGraph agent result."""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        messages = result.get("messages") or []
        if messages:
            content = getattr(messages[-1], "content", None)
            if isinstance(content, str):
                return content
            return str(content)
        if "structured_response" in result:
            return str(result["structured_response"])

    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content

    return str(result)


def _create_agent(model, tools, system_prompt: str):
    """Create a LangChain/LangGraph tool-calling agent.

    Prefer the modern LangChain create_agent API, with a LangGraph prebuilt
    fallback for environments pinned to older package versions.
    """

    try:
        from langchain.agents import create_agent

        return create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )
    except Exception:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt,
        )


class LangGraphTripPlanner:
    """Multi-agent trip planner implemented with LangGraph."""

    def __init__(self):
        print("🔄 开始初始化 LangGraph 多智能体旅行规划系统...")
        self.llm = get_llm()
        self._compiled_graph = None
        self._agents_ready = False
        self._tools = []
        self._tool_names: List[str] = []
        self.attraction_agent = None
        self.weather_agent = None
        self.hotel_agent = None

    async def _ensure_ready(self) -> None:
        if self._agents_ready:
            return

        self._tools = await get_amap_tools()
        self._tool_names = [tool.name for tool in self._tools]

        attraction_tools = filter_tools(self._tools, "maps_text_search", "text_search")
        weather_tools = filter_tools(self._tools, "maps_weather", "weather")
        hotel_tools = filter_tools(self._tools, "maps_text_search", "text_search")

        self.attraction_agent = _create_agent(
            self.llm,
            attraction_tools or self._tools,
            self._prompt_with_tools(ATTRACTION_AGENT_PROMPT, attraction_tools or self._tools),
        )
        self.weather_agent = _create_agent(
            self.llm,
            weather_tools or self._tools,
            self._prompt_with_tools(WEATHER_AGENT_PROMPT, weather_tools or self._tools),
        )
        self.hotel_agent = _create_agent(
            self.llm,
            hotel_tools or self._tools,
            self._prompt_with_tools(HOTEL_AGENT_PROMPT, hotel_tools or self._tools),
        )

        self._compiled_graph = self._build_graph()
        self._agents_ready = True
        print("✅ LangGraph 多智能体系统初始化成功")
        print(f"   MCP工具数量: {len(self._tool_names)}")

    @staticmethod
    def _prompt_with_tools(base_prompt: str, tools) -> str:
        tool_lines = "\n".join(f"- {tool.name}: {getattr(tool, 'description', '')}" for tool in tools)
        return f"{base_prompt}\n\n可用工具:\n{tool_lines}"

    def _build_graph(self):
        graph = StateGraph(TripPlannerState)
        graph.add_node("search_attractions", self._search_attractions)
        graph.add_node("query_weather", self._query_weather)
        graph.add_node("search_hotels", self._search_hotels)
        graph.add_node("compose_plan", self._compose_plan)
        graph.add_node("validate_plan", self._validate_plan)
        graph.add_node("fallback_plan", self._fallback_plan_node)

        graph.add_edge(START, "search_attractions")
        graph.add_edge(START, "query_weather")
        graph.add_edge(START, "search_hotels")
        graph.add_edge(
            ["search_attractions", "query_weather", "search_hotels"],
            "compose_plan",
        )
        graph.add_edge("compose_plan", "validate_plan")
        graph.add_conditional_edges(
            "validate_plan",
            self._route_after_validation,
            {
                "done": END,
                "fallback": "fallback_plan",
            },
        )
        graph.add_edge("fallback_plan", END)
        return graph.compile()

    async def aplan_trip(self, request: TripRequest) -> TripPlan:
        """Generate a trip plan asynchronously."""

        await self._ensure_ready()
        assert self._compiled_graph is not None

        print("\n" + "=" * 60)
        print("🚀 开始 LangGraph 多智能体协作规划旅行...")
        print(f"目的地: {request.city}")
        print(f"日期: {request.start_date} 至 {request.end_date}")
        print(f"天数: {request.travel_days}天")
        print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
        print("=" * 60 + "\n")

        result = await self._compiled_graph.ainvoke({"request": request, "errors": []})
        plan = result.get("final_plan")
        if isinstance(plan, TripPlan):
            return plan

        return self._create_fallback_plan(request)

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """Synchronous wrapper for scripts/tests outside an event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aplan_trip(request))

        raise RuntimeError("plan_trip()不能在运行中的事件循环内调用，请使用 await aplan_trip()")

    async def _search_attractions(self, state: TripPlannerState) -> Dict[str, Any]:
        request = state["request"]
        keywords = request.preferences[0] if request.preferences else "景点"
        query = (
            f"请为{request.city}搜索适合{request.travel_days}天旅行的景点。"
            f"偏好: {', '.join(request.preferences) if request.preferences else '无'}。"
            f"优先关键词: {keywords}。"
        )
        try:
            result = await self.attraction_agent.ainvoke({"messages": [HumanMessage(content=query)]})
            text = _message_text(result)
            print(f"📍 景点节点完成: {text[:200]}...")
            return {"attraction_results": text}
        except Exception as exc:
            message = f"景点搜索失败: {exc}"
            print(f"❌ {message}")
            return {"attraction_results": "", "errors": [message]}

    async def _query_weather(self, state: TripPlannerState) -> Dict[str, Any]:
        request = state["request"]
        query = f"请查询{request.city}在{request.start_date}至{request.end_date}附近的天气信息。"
        try:
            result = await self.weather_agent.ainvoke({"messages": [HumanMessage(content=query)]})
            text = _message_text(result)
            print(f"🌤️ 天气节点完成: {text[:200]}...")
            return {"weather_results": text}
        except Exception as exc:
            message = f"天气查询失败: {exc}"
            print(f"❌ {message}")
            return {"weather_results": "", "errors": [message]}

    async def _search_hotels(self, state: TripPlannerState) -> Dict[str, Any]:
        request = state["request"]
        query = (
            f"请为{request.city}{request.travel_days}天行程搜索或推荐{request.accommodation}。"
            f"交通偏好: {request.transportation}。"
        )
        try:
            result = await self.hotel_agent.ainvoke({"messages": [HumanMessage(content=query)]})
            text = _message_text(result)
            print(f"🏨 酒店节点完成: {text[:200]}...")
            return {"hotel_results": text}
        except Exception as exc:
            message = f"酒店搜索失败: {exc}"
            print(f"❌ {message}")
            return {"hotel_results": "", "errors": [message]}

    async def _compose_plan(self, state: TripPlannerState) -> Dict[str, Any]:
        request = state["request"]
        prompt = self._build_planner_query(
            request=request,
            attractions=state.get("attraction_results", ""),
            weather=state.get("weather_results", ""),
            hotels=state.get("hotel_results", ""),
            errors=state.get("errors", []),
        )

        try:
            structured_llm = self.llm.with_structured_output(TripPlan)
            plan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            if isinstance(plan, TripPlan):
                print("📋 行程规划节点完成: 已生成 TripPlan")
                return {"final_plan": plan}
            return {"errors": [f"结构化输出类型异常: {type(plan).__name__}"]}
        except Exception as exc:
            message = f"行程规划失败: {exc}"
            print(f"❌ {message}")
            return {"errors": [message]}

    def _validate_plan(self, state: TripPlannerState) -> Dict[str, Any]:
        plan = state.get("final_plan")
        request = state["request"]
        errors = []

        if not isinstance(plan, TripPlan):
            errors.append("未生成有效 TripPlan")
        else:
            if plan.city != request.city:
                errors.append("城市与请求不一致")
            if len(plan.days) != request.travel_days:
                errors.append("行程天数与请求不一致")
            for day in plan.days:
                if len(day.meals) < 3:
                    errors.append(f"{day.date} 餐饮信息不足")
                if not day.attractions:
                    errors.append(f"{day.date} 景点信息为空")

        if errors:
            print(f"⚠️  校验发现问题: {'; '.join(errors)}")
            return {"errors": errors}

        print("✅ 行程计划校验通过")
        return {}

    @staticmethod
    def _route_after_validation(state: TripPlannerState) -> str:
        plan = state.get("final_plan")
        if isinstance(plan, TripPlan):
            request = state["request"]
            if plan.city == request.city and len(plan.days) == request.travel_days:
                return "done"
        return "fallback"

    def _fallback_plan_node(self, state: TripPlannerState) -> Dict[str, Any]:
        print("🧯 使用备用方案生成计划")
        return {"final_plan": self._create_fallback_plan(state["request"])}

    @staticmethod
    def _build_planner_query(
        request: TripRequest,
        attractions: str,
        weather: str,
        hotels: str,
        errors: Optional[List[str]] = None,
    ) -> str:
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

基本信息:
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

景点信息:
{attractions or '无可用景点工具结果'}

天气信息:
{weather or '无可用天气工具结果'}

酒店信息:
{hotels or '无可用酒店工具结果'}
"""
        if request.free_text_input:
            query += f"\n额外要求: {request.free_text_input}\n"
        if errors:
            query += "\n上游节点问题:\n" + "\n".join(f"- {error}" for error in errors)
        return query

    @staticmethod
    def _create_fallback_plan(request: TripRequest) -> TripPlan:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        days = []

        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i + 1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j + 1}",
                        address=f"{request.city}市",
                        location=Location(
                            longitude=116.4 + i * 0.01 + j * 0.005,
                            latitude=39.9 + i * 0.01 + j * 0.005,
                        ),
                        visit_duration=120,
                        description=f"这是{request.city}的候选景点，请以实际开放信息为准。",
                        category="景点",
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i + 1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i + 1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i + 1}天晚餐", description="晚餐推荐"),
                ],
            )
            days.append(day_plan)

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=(
                f"这是为您规划的{request.city}{request.travel_days}日游备用行程。"
                "建议提前查看景点开放时间、天气和交通信息。"
            ),
        )


_trip_planner: LangGraphTripPlanner | None = None


def get_trip_planner_agent() -> LangGraphTripPlanner:
    """Return singleton planner instance."""

    global _trip_planner

    if _trip_planner is None:
        _trip_planner = LangGraphTripPlanner()

    return _trip_planner
