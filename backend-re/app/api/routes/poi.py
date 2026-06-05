"""POI helper routes for frontend compatibility."""

from fastapi import APIRouter


router = APIRouter(prefix="/poi", tags=["POI"])


@router.get(
    "/photo",
    summary="获取景点图片",
    description="兼容前端图片请求。当前重写后端暂不接入图片服务，返回空图片地址。",
)
async def get_attraction_photo(name: str):
    return {
        "success": True,
        "message": "暂无图片服务，前端将使用占位图",
        "data": {
            "name": name,
            "photo_url": None,
        },
    }
