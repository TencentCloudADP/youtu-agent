"""
tione env: https://iwiki.woa.com/p/4015475657
cecret key: https://cloud.tencent.com/document/product/1278/85305
"""

import asyncio
import os
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from aiolimiter import AsyncLimiter
    from tencentcloud.common import credential
    from tencentcloud.common.common_client import CommonClient
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile


class TioneEnvManager:
    def __init__(
        self, type: Literal["OS", "BROWSER_QB", "BROWSER_CHROMIUM"] = "BROWSER_CHROMIUM", concurrency: int = 20
    ):
        self.type = type
        cred = credential.Credential(os.getenv("TENCENTCLOUD_SECRET_ID"), os.getenv("TENCENTCLOUD_SECRET_KEY"))
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tione.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.client = CommonClient("tione", "2021-11-11", cred, "ap-shanghai", profile=clientProfile)

        self.limiter = AsyncLimiter(max_rate=concurrency, time_period=1)

    def get_params(self):
        if self.type == "BROWSER_QB":
            return {
                "EnvType": "BROWSER_QB",
                "ResourceGroupId": os.getenv("TENCENTCLOUD_RESOURCE_GROUP_ID"),
                "ResourceInfo": {"Cpu": 2000, "Memory": 4000},
                "Env": [{"Name": "POD_IP", "Value": "0.0.0.0"}],
            }
        elif self.type == "BROWSER_CHROMIUM":
            return {
                "EnvType": "BROWSER_CHROMIUM",
                "ResourceGroupId": os.getenv("TENCENTCLOUD_RESOURCE_GROUP_ID"),
                "ResourceInfo": {"Cpu": 2000, "Memory": 4000},
                "VolumeMount": {
                    "VolumeSourceType": "CFS",
                    "CFSConfig": {
                        "Id": "cfs-4c1cf8410",
                        "Path": "/easonsshi/Env/browser_sessions/",
                        "MountType": "SOURCE",
                    },
                },
                "Env": [{"Name": "ENV", "Value": "tione"}],
            }
        else:
            raise ValueError(f"Unknown type: {self.type}")

    def client_call_json(self, action: str, params: dict) -> dict:
        # print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')}: {action}")
        return self.client.call_json(action, params)

    # NOTE: 原生的API调用延迟有点高, 所以改成后台异步
    async def call_json(self, action: str, params: dict) -> dict:
        async with self.limiter:
            return await asyncio.to_thread(self.client_call_json, action, params)

    async def create_env(self) -> dict:
        # 1. CreateAgentToolEnv
        params = self.get_params()
        res = await self.call_json("CreateAgentToolEnv", params)
        env_info = res["Response"]["AgentToolEnvInfo"]
        # print(f"env_info: {env_info}")
        while env_info["Status"] in ("", "CREATED"):
            await asyncio.sleep(0.5)
            env_info = await self.describe_env(env_info["EnvId"])
            # print(f"Status: {env_info['Status']}")
        return env_info

    async def describe_envs(self) -> list[dict]:
        # 2. DescribeAgentToolEnvs
        res = await self.call_json("DescribeAgentToolEnvs", {"Offset": 0})
        total_num = res["Response"]["TotalCount"]
        print(f"Number of instances: {total_num}")
        infos: list[dict] = res["Response"]["AgentToolEnvInfos"]
        for i in range(20, total_num, 20):  # 20 items per page
            res = await self.call_json("DescribeAgentToolEnvs", {"Offset": i})
            infos.extend(res["Response"]["AgentToolEnvInfos"])
        assert len({info["EnvId"] for info in infos}) == len(infos)
        assert len(infos) == total_num
        print(f"received {len(infos)} instances")
        return infos

    async def describe_env(self, env_id: str) -> dict:
        # 3. DescribeAgentToolInfoByEnvId
        res = await self.call_json("DescribeAgentToolInfoByEnvId", {"EnvId": env_id})
        return res["Response"]["AgentToolEnvInfo"]

    async def delete_env(self, env_id: str):
        # 4. DeleteAgentToolEnv
        res = await self.call_json("DeleteAgentToolEnv", {"EnvId": env_id})
        print(res)
