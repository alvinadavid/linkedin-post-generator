import json
from langchain_core.exceptions import OutputParserException
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means Hindi + English)

    Here is the actual post on which you need to perform this task:  
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})

    try:
        # Ensure the response is valid JSON
        response_content = response.content.strip()
        res = json.loads(response_content)
        # Validate the structure
        if all(key in res for key in ["line_count", "language", "tags"]):
            return res
        else:
            raise ValueError("Missing required keys in the LLM response.")
    except (json.JSONDecodeError, ValueError, OutputParserException) as e:
        print(f"Error processing post: {e}")
        return {"line_count": 0, "language": "Unknown", "tags": []}
