import json
from openai import OpenAI, pydantic_function_tool
import requests
from pydantic import BaseModel, Field


def handle_tool_call(completion, messages, client, tools):
    if completion.choices[0].message.tool_calls:
        messages.append(completion.choices[0].message)

        for tool_call in completion.choices[0].message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"Tool call in response: {name} with args: {args}")
            result = call_function(name, args)
            print(f"Tool call result: {result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

        return client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
    else:
        return completion


def get_weather_temperature_in_fahrenheit(latitude, longitude):
    print(f"Tool call: get_weather({latitude}, {longitude})")
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return convert_celsius_to_fahrenheit(data["current"]["temperature_2m"])


def add_two_numbers(a, b):
    return a + b


def convert_celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32


def call_function(name, args):
    if name == "GetWeatherInFahrenheit":
        return get_weather_temperature_in_fahrenheit(**args)
    if name == "AddTwoNumbers":
        return add_two_numbers(**args)

    return "Not implemented"


def main():
    import sys

    user_input = ""

    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        print(user_input)
    else:
        print("Hello - no params")
    client = OpenAI()

    class GetWeatherInFahrenheit(BaseModel):
        latitude: float = Field(..., description="Latitude coordinate")
        longitude: float = Field(..., description="Longitude coordinate")

        model_config = {"strict": True, "extra": "forbid"}

    class AddTwoNumbers(BaseModel):
        a: float = Field(..., description="First number")
        b: float = Field(..., description="Second number")

        model_config = {"strict": True, "extra": "forbid"}

    tools = [
        pydantic_function_tool(GetWeatherInFahrenheit),
        pydantic_function_tool(AddTwoNumbers),
    ]

    messages = [
        {
            "role": "user",
            "content": user_input,
        }
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    completion = handle_tool_call(completion, messages, client, tools)

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
