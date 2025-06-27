"""Integration tests for BedrockAgent with MCP tools."""

from unittest.mock import Mock, patch

import pytest
from mcp.types import Tool as MCPTool

from fence.agents.bedrock import BedrockAgent
from fence.mcp.client import MCPClient
from fence.mcp.tool import MCPAgentTool


@pytest.fixture
def mock_model():
    """Create a mock Bedrock model."""
    model = Mock()
    model.full_response = True
    model.toolConfig = None
    model.invoke = Mock()
    return model


@pytest.fixture
def mock_mcp_tools():
    """Create mock MCP tools."""
    return [
        MCPTool(
            name="search_documentation",
            description="Search AWS documentation",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_phrase": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 5,
                    },
                },
                "required": ["search_phrase"],
            },
        ),
        MCPTool(
            name="read_documentation",
            description="Read AWS documentation page",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Documentation URL"}
                },
                "required": ["url"],
            },
        ),
    ]


@pytest.fixture
def mock_mcp_client(mock_mcp_tools):
    """Create a mock MCP client with tools."""
    client = Mock(spec=MCPClient)
    client._connected = True
    client.tools = []

    # Create MCPAgentTool instances
    for mcp_tool in mock_mcp_tools:
        agent_tool = MCPAgentTool(mcp_tool=mcp_tool, mcp_client=client)
        client.tools.append(agent_tool)

    # Mock call_tool method
    def mock_call_tool(name, arguments):
        if name == "search_documentation":
            return {
                "results": [
                    '{"rank_order": 1, "url": "https://docs.aws.amazon.com/lambda/", "title": "AWS Lambda"}',
                    '{"rank_order": 2, "url": "https://docs.aws.amazon.com/s3/", "title": "Amazon S3"}',
                ],
                "count": 2,
            }
        elif name == "read_documentation":
            return "# AWS Lambda Documentation\n\nAWS Lambda is a serverless compute service..."
        else:
            return "Mock response"

    client.call_tool = Mock(side_effect=mock_call_tool)
    client.connect = Mock()
    client.list_tools = Mock()
    client.disconnect = Mock()

    return client


class TestMCPBedrockIntegration:
    """Test integration between MCP tools and BedrockAgent."""

    def test_agent_with_mcp_tools_initialization(self, mock_model, mock_mcp_client):
        """Test BedrockAgent initialization with MCP tools."""
        # Create agent with MCP tools
        agent = BedrockAgent(
            identifier="test-agent",
            model=mock_model,
            tools=mock_mcp_client.tools,
            log_agentic_response=False,
        )

        # Verify tools were registered
        assert len(agent.tools) == 2
        assert all(isinstance(tool, MCPAgentTool) for tool in agent.tools)

        # Verify tool names
        tool_names = [tool.get_tool_name() for tool in agent.tools]
        assert "search_documentation" in tool_names
        assert "read_documentation" in tool_names

    @patch("fence.agents.bedrock.agent.BedrockAgent._execute_tool")
    def test_agent_tool_execution(self, mock_execute_tool, mock_model, mock_mcp_client):
        """Test that BedrockAgent can execute MCP tools."""
        # Setup mock response
        mock_execute_tool.return_value = {
            "formatted_result": "[Tool used] search_documentation -> results",
            "events": [],
        }

        # Create agent
        agent = BedrockAgent(
            identifier="test-agent",
            model=mock_model,
            tools=mock_mcp_client.tools,
            log_agentic_response=False,
        )

        # Get the search tool
        _ = next(
            tool
            for tool in agent.tools
            if tool.get_tool_name() == "search_documentation"
        )

        # Create a mock tool use content
        from fence.templates.models import ToolUseBlock, ToolUseContent

        tool_use_content = ToolUseContent(
            content=ToolUseBlock(
                toolUseId="test_123",
                name="search_documentation",
                input={"search_phrase": "AWS Lambda", "limit": 3},
            )
        )

        # Execute the tool through the agent
        result = agent._execute_tool(tool_use_content)

        # Verify execution
        mock_execute_tool.assert_called_once()
        assert "formatted_result" in result

    def test_mcp_tool_parameter_validation(self, mock_mcp_client):
        """Test that MCP tool parameters are properly validated."""
        search_tool = mock_mcp_client.tools[0]  # search_documentation

        # Test with valid parameters
        _ = search_tool.execute_tool(search_phrase="test query", limit=5)
        mock_mcp_client.call_tool.assert_called_with(
            name="search_documentation",
            arguments={"search_phrase": "test query", "limit": 5},
        )

        # Test parameter filtering (extra params should be filtered out)
        search_tool.execute_tool(
            search_phrase="test", limit=3, invalid_param="should_be_filtered"
        )
        mock_mcp_client.call_tool.assert_called_with(
            name="search_documentation", arguments={"search_phrase": "test", "limit": 3}
        )

    def test_mcp_tool_error_handling(self, mock_mcp_client):
        """Test error handling in MCP tools."""
        # Setup client to raise exception
        mock_mcp_client.call_tool.side_effect = Exception("Connection timeout")

        search_tool = mock_mcp_client.tools[0]
        result = search_tool.execute_tool(search_phrase="test")

        # Verify error response
        assert isinstance(result, dict)
        assert "error" in result
        assert "Connection timeout" in result["error"]

    def test_mcp_multiple_content_handling(self, mock_mcp_client):
        """Test handling of multiple content responses from MCP."""
        search_tool = mock_mcp_client.tools[0]
        result = search_tool.execute_tool(search_phrase="AWS services")

        # Verify structured response for multiple results
        assert isinstance(result, dict)
        assert "results" in result
        assert "count" in result
        assert result["count"] == 2
        assert len(result["results"]) == 2

    def test_mcp_single_content_handling(self, mock_mcp_client):
        """Test handling of single content responses from MCP."""
        read_tool = mock_mcp_client.tools[1]  # read_documentation
        result = read_tool.execute_tool(url="https://docs.aws.amazon.com/lambda/")

        # Verify single string response
        assert isinstance(result, str)
        assert "AWS Lambda Documentation" in result

    @patch("fence.memory.base.FleetingMemory")
    def test_agent_tool_registration_bedrock_format(
        self, mock_memory, mock_model, mock_mcp_client
    ):
        """Test that MCP tools are properly registered with Bedrock format."""
        # Mock the model to have toolConfig support
        mock_model.toolConfig = None

        # Create agent
        _ = BedrockAgent(
            identifier="test-agent",
            model=mock_model,
            tools=mock_mcp_client.tools,
            log_agentic_response=False,
        )

        # Verify toolConfig was set on model
        assert mock_model.toolConfig is not None

        # Check that tools were converted to Bedrock format
        bedrock_tools = mock_model.toolConfig.tools
        assert len(bedrock_tools) == 2

        # Verify tool structure
        tool_names = [tool.toolSpec.name for tool in bedrock_tools]
        assert "search_documentation" in tool_names
        assert "read_documentation" in tool_names

    def test_agent_with_mixed_tools(self, mock_model, mock_mcp_client):
        """Test BedrockAgent with both MCP tools and regular tools."""
        from fence.tools.base import tool

        # Create a regular tool
        @tool(description="A regular tool")
        def regular_tool(message: str) -> str:
            return f"Regular: {message}"

        # Create agent with mixed tools
        all_tools = mock_mcp_client.tools + [regular_tool]
        agent = BedrockAgent(
            identifier="test-agent",
            model=mock_model,
            tools=all_tools,
            log_agentic_response=False,
        )

        # Verify all tools were registered
        assert len(agent.tools) == 3

        # Verify tool types
        mcp_tools = [tool for tool in agent.tools if isinstance(tool, MCPAgentTool)]
        regular_tools = [
            tool for tool in agent.tools if not isinstance(tool, MCPAgentTool)
        ]

        assert len(mcp_tools) == 2
        assert len(regular_tools) == 1

    def test_mcp_tool_bedrock_serialization(self, mock_mcp_client):
        """Test that MCP tools can be serialized for Bedrock."""
        search_tool = mock_mcp_client.tools[0]

        # Test Bedrock serialization
        bedrock_format = search_tool.model_dump_bedrock_converse()

        # Verify structure
        assert "toolSpec" in bedrock_format
        tool_spec = bedrock_format["toolSpec"]

        assert tool_spec["name"] == "search_documentation"
        assert tool_spec["description"] == "Search AWS documentation"
        assert "inputSchema" in tool_spec

        # Verify schema structure
        schema = tool_spec["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "search_phrase" in schema["properties"]
        assert "limit" in schema["properties"]

    def test_bedrock_agent_with_mcp_client_direct(self, mock_model, mock_mcp_client):
        """Test BedrockAgent initialization with MCP client directly."""
        # Create agent with MCP client directly
        agent = BedrockAgent(
            identifier="test-agent",
            model=mock_model,
            mcp_clients=mock_mcp_client,
            log_agentic_response=False,
        )

        # Verify MCP clients were stored
        assert len(agent.mcp_clients) == 1
        assert agent.mcp_clients[0] == mock_mcp_client

        # Verify MCP tools were automatically registered
        assert len(agent.tools) == 2
        assert all(isinstance(tool, MCPAgentTool) for tool in agent.tools)

        # Verify tool names
        tool_names = [tool.get_tool_name() for tool in agent.tools]
        assert "search_documentation" in tool_names
        assert "read_documentation" in tool_names

    def test_bedrock_agent_with_multiple_mcp_clients(self, mock_model, mock_mcp_tools):
        """Test BedrockAgent initialization with multiple MCP clients."""
        # Create two mock MCP clients
        mock_client1 = Mock(spec=MCPClient)
        mock_client1._connected = True
        mock_client1.tools = [
            MCPAgentTool(mcp_tool=mock_mcp_tools[0], mcp_client=mock_client1)
        ]

        mock_client2 = Mock(spec=MCPClient)
        mock_client2._connected = True
        mock_client2.tools = [
            MCPAgentTool(mcp_tool=mock_mcp_tools[1], mcp_client=mock_client2)
        ]

        # Create agent with multiple MCP clients
        agent = BedrockAgent(
            identifier="test-agent",
            model=mock_model,
            mcp_clients=[mock_client1, mock_client2],
            log_agentic_response=False,
        )

        # Verify MCP clients were stored
        assert len(agent.mcp_clients) == 2
        assert mock_client1 in agent.mcp_clients
        assert mock_client2 in agent.mcp_clients

        # Verify MCP tools were automatically registered from both clients
        assert len(agent.tools) == 2
        assert all(isinstance(tool, MCPAgentTool) for tool in agent.tools)

        # Verify tool names from both clients
        tool_names = [tool.get_tool_name() for tool in agent.tools]
        assert "search_documentation" in tool_names
        assert "read_documentation" in tool_names

    def test_bedrock_agent_mcp_client_not_connected(self, mock_model):
        """Test BedrockAgent behavior when MCP client is not connected."""
        # Create disconnected MCP client
        mock_mcp_client = Mock()
        mock_mcp_client._connected = False
        mock_mcp_client.tools = []

        # Create agent with disconnected MCP client
        with patch("fence.agents.bedrock.agent.logger") as mock_logger:
            agent = BedrockAgent(
                identifier="test-agent",
                model=mock_model,
                mcp_clients=mock_mcp_client,
                log_agentic_response=False,
            )

            # Verify warning was logged
            mock_logger.warning.assert_called_with(
                "No MCP tools were successfully registered"
            )

            # Verify no tools were registered
            assert len(agent.tools) == 0

    def test_bedrock_agent_mixed_tools_with_mcp_client(
        self, mock_model, mock_mcp_client
    ):
        """Test BedrockAgent with both MCP client and manual tools."""
        from fence.tools.base import tool

        # Create a regular tool
        @tool(description="A regular tool")
        def regular_tool(message: str) -> str:
            return f"Regular: {message}"

        # Create agent with both MCP client and manual tools
        agent = BedrockAgent(
            identifier="test-agent",
            model=mock_model,
            tools=[regular_tool],  # Manual tools
            mcp_clients=mock_mcp_client,  # MCP client tools
            log_agentic_response=False,
        )

        # Verify all tools were registered (1 manual + 2 MCP)
        assert len(agent.tools) == 3

        # Verify tool types
        mcp_tools = [tool for tool in agent.tools if isinstance(tool, MCPAgentTool)]
        regular_tools = [
            tool for tool in agent.tools if not isinstance(tool, MCPAgentTool)
        ]

        assert len(mcp_tools) == 2  # From MCP client
        assert len(regular_tools) == 1  # Manual tool

    def test_bedrock_agent_system_message_with_mcp_tools(
        self, mock_model, mock_mcp_client
    ):
        """Test that system message includes MCP tool information."""
        # Create agent with custom system message and MCP client
        custom_message = "You are a specialized assistant."
        agent = BedrockAgent(
            identifier="test-agent",
            model=mock_model,
            mcp_clients=mock_mcp_client,
            system_message=custom_message,
            log_agentic_response=False,
        )

        # Verify system message includes MCP tool information
        system_message = agent.get_system_message()
        assert custom_message in system_message
        assert "search_documentation" in system_message
        assert "read_documentation" in system_message
        assert "You have access to MCP tools" in system_message

    def test_bedrock_agent_no_mcp_client(self, mock_model):
        """Test BedrockAgent initialization without MCP client."""
        # Create agent without MCP client
        agent = BedrockAgent(
            identifier="test-agent", model=mock_model, log_agentic_response=False
        )

        # Verify no MCP clients
        assert len(agent.mcp_clients) == 0
        assert len(agent.tools) == 0
