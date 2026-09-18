## Import the necessary modules
import json
import os

import ollama

## Import the function from the module parse_data
from parse_data import get_unclaimed_items, load_items, save_result


## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
## You may need to use json.dumps() to convert the available_items list into a JSON string.

def build_prompt(description, available_items):
    system_prompt = """You are a campus lost-and-found matching assistant.
Use only the items in the JSON provided by the user.
Not all details of an item must match for it to be a possible match.
Return only valid JSON with exactly this structure:
{
    \"matches\": [\"ITEM_ID\"],
    \"confidence\": \"LOW\"
}
The matches list must contain every possible matching item ID.
The confidence value must be exactly LOW, MEDIUM, or HIGH.
If there is no possible match, return an empty matches list.
Do not include explanations or Markdown in your response."""

    user_prompt = "Lost item description:\n"
    user_prompt += description
    user_prompt += "\n\nAvailable items:\n"
    user_prompt += json.dumps(available_items, ensure_ascii=False, indent=2)

    return system_prompt, user_prompt
    

## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result. 
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    if not isinstance(response_text, str):
        raise ValueError("The model response must be a string.")

    response_text = response_text.strip()
    if response_text.startswith("```") and response_text.endswith("```"):
        lines = response_text.splitlines()
        response_text = "\n".join(lines[1:-1]).strip()

    return json.loads(response_text)
    


## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence", and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs .
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False

    if "matches" not in result or "confidence" not in result:
        return False

    if not isinstance(result["matches"], list):
        return False

    if not isinstance(result["confidence"], str):
        return False

    if result["confidence"] not in {"LOW", "MEDIUM", "HIGH"}:
        return False

    valid_ids = {item.get("id") for item in available_items}
    return all(
        isinstance(item_id, str) and item_id in valid_ids
        for item_id in result["matches"]
    )


## Logic to display the matches found by Qwen in a user-friendly format.
## It should look something like this:
""" 
CAMPUS LOST-AND-FOUND ASSISTANT
==================================================

Describe the item you lost: I lost a black bag somewhere

Searching for possible matches...

MATCH RESULT
--------------------------------------------------
Confidence: MEDIUM

Possible matches:

ID: F101
Item: backpack
Color: black
Location: Library 2nd floor
Date found: 2026-09-15

Result saved to output/match_result.json
 """
## If no matches are found, it should display a message indicating that no matches were found, along with the empty list
def display_matches(result, available_items):
    print("\nMATCH RESULT")
    print("-" * 50)
    print(f"Confidence: {result['confidence']}")

    if not result["matches"]:
        print("\nNo possible matches were found.")
        print("Matches: []")
        return

    items_by_id = {item.get("id"): item for item in available_items}
    print("\nPossible matches:")
    for item_id in result["matches"]:
        item = items_by_id[item_id]
        print(f"\nID: {item.get('id')}")
        print(f"Item: {item.get('item')}")
        print(f"Color: {item.get('color')}")
        print(f"Location: {item.get('location')}")
        print(f"Date found: {item.get('date')}")
    

## Control center for the entire program.
def main():
    items = load_items("found_items.json")
    available_items = get_unclaimed_items(items)

    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50)
    description = input("\nDescribe the item you lost: ")

    print("\nSearching for possible matches...")
    system_prompt, user_prompt = build_prompt(description, available_items)

    try:
        response_text = ask_qwen(system_prompt, user_prompt)
        result = parse_response(response_text)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"\nUnable to process the model response: {error}")
        return None

    if not validate_result(result, available_items):
        print("\nThe model returned an invalid result.")
        return None

    display_matches(result, available_items)
    output_file = "output/match_result.json"
    save_result(result, output_file)
    print(f"\nResult saved to {output_file}")
    return result


if __name__ == "__main__":
    main()
