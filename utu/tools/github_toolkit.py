"""
https://docs.github.com/en/rest?apiVersion=2022-11-28
"""

from urllib.parse import urlparse

import requests

from ..utils import EnvUtils
from .base import AsyncBaseToolkit, register_tool


class GitHubToolkit(AsyncBaseToolkit):
    """GitHub Toolkit

    - API key: `GITHUB_TOKEN`
    """

    @register_tool
    async def get_repo_info(self, github_url) -> dict:
        """Get the info of the specified github repo

        Args:
            github_url (str): The url to get content from.

        Returns:
            dict: The info of the specified github repo"""
        parsed_url = urlparse(github_url)
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            return {"error": "Invalid GitHub repository URL"}
        api_url = f"https://api.github.com/repos/{path_parts[0]}/{path_parts[1]}"
        headers = {
            "Authorization": f"Bearer {EnvUtils.get_env('GITHUB_TOKEN')}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            repo_data = response.json()
            assert repo_data is not None, f"Failed to get repository info: {response.text}\nurl:{github_url}"
            license_info = repo_data.get("license")
            license_name = license_info.get("name") if license_info else "Not specified"
            info = {
                "name": repo_data["name"],
                "owner": repo_data["owner"]["login"],
                "star": repo_data["stargazers_count"],
                "fork": repo_data["forks_count"],
                "watcher": repo_data["watchers_count"],
                "license": license_name,
                "language": repo_data.get("language", "Not specified"),
                "created_at": repo_data["created_at"],
                "updated_at": repo_data["updated_at"],
                "description": repo_data.get("description", "No description"),
                "open_issues_count": repo_data["open_issues_count"],
                "html_url": repo_data["html_url"],
            }
            return info
        except requests.exceptions.RequestException as e:
            return {"error": f"Request error: {str(e)}"}
        except (ValueError, KeyError) as e:
            return {"error": f"Failed to parse data: {str(e)}"}
