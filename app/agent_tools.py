"""Shared tool contracts; chat and research remain independent business adapters."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class QueryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    phone_name: str = Field(min_length=1, max_length=100, description="本次候选中的完整机型名称")
    query: str = Field(min_length=1, max_length=200, description="自行决定的检索词；公开搜索须包含完整机型")
    backend: Literal["sogou", "bing", "google"] = Field(default="sogou", description="公开搜索引擎；优先sogou，结果不相关时自主换引擎。本地检索忽略")


class LocalQueryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    phone_name: str = Field(min_length=1, max_length=100, description="本地已有资料的完整机型名")
    query: str = Field(min_length=1, max_length=200, description="只检索本地资料的主题；此工具不联网")


class LocalDiscoveryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=200, description="按本轮用途与需求在本地资料跨机型检索，不必先猜型号，不联网")
    limit: int = Field(default=5, ge=1, le=5)


class FetchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    phone_name: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=1, max_length=2000, description="此前为同款机型搜索返回的URL")
    focus: str | None = Field(default=None, min_length=1, max_length=200, description="本页重点提取什么，如相机配置、版本价格、夜景样张；不填则沿用当前需求")


class DiscoveryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=200, description="按预算价位和用途搜索新机型的查询词，不必指定型号")
    url: str | None = Field(default=None, max_length=2000, description="可选：打开先前discover_phones返回的公开目录或文章URL寻找型号；不填则搜索")
    backend: Literal["sogou", "bing", "google"] = Field(default="sogou", description="自行选择引擎，优先sogou，返回无关结果时可换引擎")


class CandidateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    names: list[str] = Field(min_length=1, max_length=3, description="提出待验证的完整型号；不代表存在、上市或满足任何条件，须继续查正文")


TOOL_DEFINITIONS = {
    "search_local_phones": (LocalDiscoveryArguments, "在本地知识库按意思寻找多款候选并取正文，优先复用现有资料，无搜索或聊天API请求；结果仍需核对价位与条件。"),
    "consider_phones": (CandidateArguments, "登记你提出的待核对型号假设，使其可被公开搜索。不限本地目录，最多三款；不提供任何预设事实，最终推荐须正文、版本价格和用途证据。"),
    "discover_phones": (DiscoveryArguments, "按你决定的查询词搜索新候选，也可用url打开本工具先前返回的公开目录或文章继续寻找型号。返回线索，非已核对事实。结果不佳时自行换词、换来源或停止。"),
    "search_phone_knowledge": (LocalQueryArguments, "只检索本地已有资料，不联网。没有本地资料的型号应选web_search，不在此工具重复尝试。"),
    "web_search": (QueryArguments, "按你决定的查询词搜索公开网页；返回的是未核对正文的摘要，可继续选择抓取正文。"),
    "fetch_page": (FetchArguments, "读取此前搜索到的同机型页面；在一次有界读取内解析HTTP与搜索meta跳转并提取匹配正文，保留最终URL、跳转轨迹及失败。不执行脚本或自动打开普通链接。"),
}
