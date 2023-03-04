import openai
import os
import pdbtest

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
# Set up OpenAI API credentials
openai.api_key = open_file('openaiapikey.txt')

# Define role
role = {
    "persona": {
        "name": "Your Personal Assistant",
        "facts": [
            {"name": "Name", "value": "YOUR NAME HERE"},
            {"name": "Age", "value": "YOUR AGE HERE"},
            {"name": "Location", "value": "YOUR LOCATION HERE"},
            # Add any additional facts about yourself here
        ]
    },
    "history": []
}

# Start chat
print(f"{role['persona']['name']}: Hi there! I'm here to answer any questions you have about me. What would you like to know?")
messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
while True:
    # Get user input
    user_input = input("You: ")

    # Add user input to history
    role["history"].append({"role": "user", "content": user_input})

    # Generate response using OpenAI API
    
    # messages.append(user_input)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    pdb.set_trace()
    # Get chatbot response
    chatbot_response = response.choices[0].text.strip()

    # Add chatbot response to history
    role["history"].append({"role": "assistant", "content": chatbot_response})

    # Print chatbot response
    print(f"{role['persona']['name']}: {chatbot_response}")
