"""Sign in to Atlassian and print one Jira issue."""

import asyncio
import json
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

PROJECT_FOLDER = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_FOLDER / ".env")

MCP_URL = os.getenv("MCP_BASE_URL", "https://mcp.atlassian.com/v2/mcp")
SITE_URL = os.getenv("MCP_SITE_URL", "https://girishgk403.atlassian.net").rstrip("/")
ISSUE_KEY = os.getenv("MCP_ISSUE_KEY", "SCRUM-3")
CALLBACK_URL = "http://localhost:3030/callback"


class TokenStorageForThisRun(TokenStorage):
    """Keep sign-in information in memory until the script exits."""

    def __init__(self):
        self.tokens = None
        self.client_info = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(
        self, client_info: OAuthClientInformationFull
    ) -> None:
        self.client_info = client_info


async def main():
    # Start a temporary local listener for Atlassian's sign-in redirect.
    login_result = asyncio.get_running_loop().create_future()

    async def handle_callback(reader, writer):
        request = (await reader.readline()).decode("ascii", errors="replace")
        parts = request.split()

        if len(parts) < 2:
            writer.close()
            await writer.wait_closed()
            return

        callback = urlparse(parts[1])
        parameters = parse_qs(callback.query)

        # Read the rest of the browser request.
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass

        if callback.path == "/callback" and "error" in parameters:
            message = "Sign-in was not approved. You can close this page."
            status = "400 Bad Request"
            if not login_result.done():
                login_result.set_exception(
                    RuntimeError(f"Atlassian sign-in failed: {parameters['error'][0]}")
                )
        elif callback.path == "/callback" and "code" in parameters:
            message = "Atlassian sign-in complete. You can close this page."
            status = "200 OK"
            if not login_result.done():
                login_result.set_result(
                    (parameters["code"][0], parameters.get("state", [None])[0])
                )
        else:
            message = "Sign-in failed. You can close this page."
            status = "400 Bad Request"
            if callback.path == "/callback" and not login_result.done():
                login_result.set_exception(
                    RuntimeError("Atlassian did not return a sign-in code.")
                )

        body = message.encode("utf-8")
        writer.write(
            (
                f"HTTP/1.1 {status}\r\n"
                "Content-Type: text/plain; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    try:
        callback_server = await asyncio.start_server(
            handle_callback, "localhost", 3030
        )
    except OSError as error:
        raise RuntimeError(
            "Could not start sign-in on localhost port 3030. "
            "Close any program using that port and try again."
        ) from error

    try:
        # These functions open the browser and wait for Atlassian to return.
        async def open_login_page(url):
            print("Opening Atlassian sign-in in your browser...")
            if not webbrowser.open(url):
                print("Open this link in your browser:")
                print(url)

        async def wait_for_login():
            return await login_result

        # Connect to Atlassian using browser sign-in.
        oauth = OAuthClientProvider(
            server_url=MCP_URL,
            client_metadata=OAuthClientMetadata(
                client_name="Jira issue reader",
                redirect_uris=[CALLBACK_URL],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
            ),
            storage=TokenStorageForThisRun(),
            redirect_handler=open_login_page,
            callback_handler=wait_for_login,
        )
        client = MultiServerMCPClient(
            {
                "atlassian": {
                    "transport": "streamable_http",
                    "url": MCP_URL,
                    "auth": oauth,
                }
            }
        )

        print("Connecting to Atlassian...")
        tools = await client.get_tools(server_name="atlassian")

        # Look up the cloud ID for the Jira site in .env.
        sites_tool = None
        for tool in tools:
            if tool.name == "getAccessibleAtlassianResources":
                sites_tool = tool
                break
        if sites_tool is None:
            raise RuntimeError("Atlassian did not provide its site lookup tool.")

        sites_result = await sites_tool.ainvoke({})
        sites = json.loads(sites_result[0]["text"])["data"]["resources"]

        site = None
        for available_site in sites:
            url = available_site.get("url", "").rstrip("/")
            if url.casefold() == SITE_URL.casefold():
                site = available_site
                break
        if site is None:
            raise RuntimeError(f"Could not find your Jira site: {SITE_URL}")

        # Find the tool that reads an issue.
        issue_tool = None
        for tool in tools:
            if tool.name == "getJiraIssue":
                issue_tool = tool
                break
        if issue_tool is None:
            raise RuntimeError("Atlassian did not provide its Jira issue tool.")

        # Read the issue and print Atlassian's response.
        print(f"Reading {ISSUE_KEY}...")
        result = await issue_tool.ainvoke(
            {
                "cloudId": site["cloudId"],
                "issueIdOrKey": ISSUE_KEY,
                "view": "evidence",
            }
        )
        issue_text = result[0]["text"]
        issue = json.loads(issue_text)
        if issue.get("error"):
            raise RuntimeError(issue.get("message", "Could not read the Jira issue."))

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
        print(issue_text)
    finally:
        callback_server.close()
        await callback_server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
