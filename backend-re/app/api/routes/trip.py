"""Trip planning routes for the LangGraph backend."""

from fastapi import APIRouter, HTTPException

from ...agents.trip_planner_graph import get_trip_planner_agent
from ...models.schemas import TripPlanResponse, TripRequest


router = APIRouter(prefix="/trip", tags=["旅行规划"])


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="基于 LangChain/LangGraph 多智能体生成详细旅行计划",
)
async def plan_trip(request: TripRequest):
    try:
        print("\n" + "=" * 60)
        print("📥 收到旅行规划请求:")
        print(f"   城市: {request.city}")
        print(f"   日期: {request.start_date} - {request.end_date}")
        print(f"   天数: {request.travel_days}")
        print("=" * 60 + "\n")

        planner = get_trip_planner_agent()
        trip_plan = await planner.aplan_trip(request)

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功",
            data=trip_plan,
        )
    except Exception as exc:
        print(f"❌ 生成旅行计划失败: {exc}")
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {exc}") from exc


@router.get(
    "/health",
    summary="健康检查",
    description="检查 LangGraph 旅行规划服务是否正常",
)
async def health_check():
    try:
        planner = get_trip_planner_agent()
        await planner._ensure_ready()
        return {
            "status": "healthy",
            "service": "langgraph-trip-planner",
            "planner": planner.__class__.__name__,
            "tools_count": len(planner._tool_names),
            "tools": planner._tool_names,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"服务不可用: {exc}") from exc
