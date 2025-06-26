from mcp import StdioServerParameters, stdio_client


def create_stdio_transport(command: str, args: list[str]):
    server_parameters = StdioServerParameters(
        command=command,
        args=args,
    )
    return stdio_client(server_parameters)
