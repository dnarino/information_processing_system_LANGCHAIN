
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Literal
from pydantic import BaseModel, Field
from pprint import pprint
from core.openai_client import get_openai_chat


# Sample product reviews for testing

reviews_dict = {
    "positive_coffee": (
        "I absolutely love this coffee maker! It brews quickly and the coffee "
        "tastes amazing. The built-in grinder saves me so much time in the "
        "morning, and the programmable timer means I wake up to fresh coffee "
        "every day. Worth every penny and highly recommended to any coffee enthusiast."
    ),
    "negative_laptop": (
        "Disappointed with this laptop. It's constantly overheating after just "
        "30 minutes of use, and the battery life is nowhere near the 8 hours "
        "advertised - I barely get 3 hours. The keyboard has already started "
        "sticking on several keys after just two weeks. Would not recommend to anyone."
    ),
    "neutral_backpack": (
        "The backpack is okay. It holds my laptop and a few books, but it lacks "
        "any special compartments for organization. The material feels durable enough, "
        "but the straps aren't very padded. It does the job for a daily commute, but "
        "it's nothing extraordinary."
    ),
    "positive_headphones": (
        "These noise-canceling headphones are a game changer! The sound quality is "
        "crystal clear, and the battery lasts for days on a single charge. I wear "
        "them in the office and can't hear a single distraction. Best purchase I've made all year."
    ),
    "negative_blender": (
        "Terrible blender. It struggles to crush ice and leaves huge chunks of fruit "
        "in my smoothies. To make matters worse, it's incredibly loud and the base "
        "started leaking after the third use. I will be returning this immediately."
    ),
    "neutral_desk": (
        "It's a standard standing desk. Assembly took about an hour, and the motor "
        "is reasonably quiet when adjusting the height. The surface looks a bit cheap "
        "and scratches easily, but it holds my monitors securely. Average value for the price."
    ),
    "positive_shoes": (
        "Most comfortable running shoes I have ever owned! I just finished a half "
        "marathon and my feet feel fantastic. The cushioning provides excellent support, "
        "and they are surprisingly lightweight. I'm definitely buying another pair."
    ),
    "negative_software": (
        "This photo editing software is a nightmare. The interface is completely "
        "unintuitive, and it constantly crashes when I try to export high-resolution "
        "images. Customer support hasn't replied to my emails in over a week."
    ),
    "neutral_vacuum": (
        "The robot vacuum is decent. It navigates around my living room fairly well, "
        "but it occasionally gets stuck under the couch. It picks up surface dust fine, "
        "but struggles with pet hair on carpets. It saves me some sweeping time, but I still have to manually vacuum once a week."
    ),
    "positive_monitor": (
        "Stunning 4K monitor! The colors are incredibly vibrant, and the 144Hz refresh "
        "rate makes gaming buttery smooth. The ultra-thin bezels look fantastic on my desk. "
        "I honestly can't believe the quality for the price I paid."
    ),
    "negative_chair": (
        "This ergonomic chair is anything but ergonomic. The lumbar support digs into "
        "my lower back aggressively, and the seat cushion flattened out completely "
        "within a month. For $300, I expected much better build quality."
    ),
    "neutral_watch": (
        "The smartwatch looks nice and tracks my steps accurately, but the battery "
        "needs to be charged every single night, which is annoying. The heart rate "
        "monitor seems reliable, though the app interface could use an update."
    )
}

# 1. Setup the OpenAI LLM
# Note: OpenAI uses max_tokens instead of max_new_tokens!
parameters = {
    "max_tokens": 1024,
    "temperature": 0.2,
    "model_kwargs": {
        "top_p": 0.1
    }
}
openai_LLM = get_openai_chat(params=parameters)

# ----------------- SENTIMENT CHAIN -----------------
class SentimentPydantic(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description='The sentiment of the review')

sentiment_parser = PydanticOutputParser(pydantic_object=SentimentPydantic)
sentiment_instructions = sentiment_parser.get_format_instructions()

sentiment_promt = PromptTemplate.from_template(
    template="""Analyze the sentiment of the following product review as positive, negative, or neutral.
    Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral] \n {sentiment_instructions}"

    Review: {review}

    Your analysis:
    """,
    partial_variables={'sentiment_instructions': sentiment_instructions}
)

sentiment_chain = sentiment_promt | openai_LLM | sentiment_parser

# ----------------- SUMMARY CHAIN -----------------
class SummaryPydantic(BaseModel):
    key_points: list[str] = Field(description="A list of 3-5 key points.")     

summary_parser = PydanticOutputParser(pydantic_object=SummaryPydantic)
summary_instructions = summary_parser.get_format_instructions()

summary_promtp = PromptTemplate.from_template(
    template="""
        Summarize the following product review into 3-5 key bullet points.
        Each bullet point should be concise and capture an important aspect mentioned in the review.\n{summary_instructions}

        Review: {review}
        Sentiment: {sentiment}

        Key points:
    """,
    partial_variables={'summary_instructions': summary_instructions}
) 

summary_chain = summary_promtp | openai_LLM | summary_parser

# ----------------- RESPONSE CHAIN -----------------

class ResponsePydantic(BaseModel):
    response: str = Field(description='The AI response to the human customer')

response_parser = PydanticOutputParser(pydantic_object=ResponsePydantic)
response_instructions = response_parser.get_format_instructions()

response_prompt = PromptTemplate.from_template(
    template="""
        Write a helpful response to a customer based on their product review.
        If the sentiment is positive, thank them for their feedback. If negative, express understanding 
        and suggest a solution or next steps. Personalize based on the specific points they mentioned.\n{response_instructions}
        Review: {review}
        Sentiment: {sentiment}
        Key points: {summary}
        Response to customer:
    """,
    partial_variables={'response_instructions': response_instructions}
)

response_chain = response_prompt | openai_LLM | response_parser

# ----------------- MASTER PIPELINE -----------------
overall_chain = (
    {"review": RunnablePassthrough()}
    | RunnablePassthrough.assign(sentiment=sentiment_chain)
    | RunnablePassthrough.assign(summary=summary_chain)
    | RunnablePassthrough.assign(response=response_chain)
)


import asyncio

async def main():
    print("Running ASYNC batch processing using OpenAI...")
    
    # We grab all the reviews and turn them into a list
    list_of_reviews = list(reviews_dict.values())
    
    # .abatch() fires off ALL reviews to OpenAI at the exact same time!
    results = await overall_chain.abatch(list_of_reviews)
    
    for result in results:
        print("\n" + "="*50)
        print("--- FULL PIPELINE RESULT ---")
        pprint(result)

if __name__== "__main__":
    # asyncio.run() creates the async environment needed to use 'await'
    asyncio.run(main())
