import openai
import os

# Initialize the OpenAI API key

openai.api_key = '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'

def generate_text(prompt):
    try:
        # Call the GPT-4 model with Completion.create
        client = openai.OpenAI(
            api_key = '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'
        )


        # assistant = openai.Assistants.create(
        #     name="Vet Diagnostic Assistant",
        #     instructions="You are a personal veterinary diagnostic assistant. Provide expert advice on veterinary diagnostics and suggest appropriate actions based on the information provided.",
        #     tools=[{"type": "code_interpreter"}, {"type": "database_access"}],
        #     model="gpt-4-1106-preview",
        # )
        assistant = client.beta.assistants.create(
            name="Vet Diagnostic Assistant",
            instructions="You are a personal veterinary diagnostic assistant. Provide expert advice on veterinary diagnostics and suggest appropriate actions based on the information provided.",
            tools=[{"type": "code_interpreter"}],
            model='gpt-3.5-turbo',
            )

        thread = client.beta.threads.create()

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Please address the user as Jane Doe. The user has a premium account.",
        )

        print("Run completed with status: " + run.status)

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)

            print("messages: ")
            for message in messages:
                assert message.content[0].type == "text"
                print({"role": message.role, "message": message.content[0].text.value})

            client.beta.assistants.delete(assistant.id)
    
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    user_input = input("Enter your prompt: ")  # Example prompt
    result = generate_text(user_input)
    print("\nGenerated Text:\n", result)


# A dog has a subcutaneous mass in the abdominal area. should we try to take a sample using a test syringe in case it's a cyst, or do an x-ray to make sure it's not intestines?