from fence import ClaudeHaiku, Link, ClaudeSonnet
from fence.parsers import TOMLParser
from fence.templates.messages import MessagesTemplate, Message
from fence.utils.optim import retry
from fence.utils.logger import setup_logging
from prompts import SYSTEM_MESSAGE
from dynamo import DynamoMemory

TABLE_NAME = "chat_memory"

#model = ClaudeHaiku(source="page-chat-test", region="us-east-1")
model = ClaudeSonnet(source="page-chat-test", region="us-east-1")

logger = setup_logging(__name__, log_level="debug", serious_mode=False)


def handler(event, context):

    # Get the message
    new_user_message = event.get("message", None)
    session_id = event.get("session_id", None)
    org_uuid = event.get("org_uuid", None)

    # Initialize memory
    memory = DynamoMemory(
        table_name=TABLE_NAME, session_id=session_id, org_uuid=org_uuid
    )

    # Apply history to last messages
    new_user_message = Message(
        role="user",
        content=new_user_message,
    )
    messages, last_state, last_assets = memory.apply_history(message=new_user_message)

    # Add system prompt
    messages.system = SYSTEM_MESSAGE

    # Get the assets
    new_assets = event.get("assets")
    if new_assets:
        last_assets = new_assets

    # Add a prefill assistant message
    messages.messages.append(
        Message(
            role="assistant",
            content="```toml",
        )
    )

    # Create MessageTemplate object
    template = MessagesTemplate(source=messages)

    # Build a Link
    link = Link(
        name="chat_link",
        template=template,
        llm=model,
        parser=TOMLParser(prefill="```toml"),
        output_key="chat_response",
    )

    @retry(max_retries=3, delay=0)
    def run_link():
        return link.run()

    response = run_link()

    # Store both messages
    memory.store_message(message=new_user_message)
    memory.store_message(
        message=Message(
            role="assistant",
            content=response["chat_response"]["message"],
        ),
        state=response["chat_response"].get("state", last_state),
        assets=last_assets,
    )

    return memory.session_id, response


if __name__ == "__main__":

    session_id = None

    message = input("Enter a message: ")
    end_conversation = False
    while message != "exit":
        session_id, response = handler(
            event={"message": message, "session_id": session_id, "org_uuid": 666},
            context={},
        )
        logger.info(f"State: {response['chat_response'].get('state', None)}")
        logger.critical(f"Response: {response['chat_response']['message']}")
        end_conversation = response["chat_response"]["endConversation"]
        if end_conversation:
            break
        message = input("Enter a message: ")
