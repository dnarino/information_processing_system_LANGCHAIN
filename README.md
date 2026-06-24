# 🧠 Information Processing System (LangChain)

An enterprise-grade, asynchronous AI pipeline built with **LangChain (LCEL)** and **OpenAI**. 

This system autonomously processes batches of customer reviews through a multi-stage reasoning pipeline. By enforcing strict deterministic JSON outputs using **Pydantic** models, it eliminates LLM hallucinations and generates structured, personalized customer service responses.

## 🚀 Key Features

* **Asynchronous Batching (`.abatch`)**: Processes massive arrays of customer reviews concurrently via `asyncio`, reducing execution time from minutes to mere seconds.
* **Modern LCEL Architecture**: Replaces legacy `SequentialChains` with a highly modular, readable, pipe-based architecture.
* **Strict State Management**: Utilizes `RunnablePassthrough.assign()` to accumulate contextual state (Sentiment -> Summary -> Response) across independent LLM execution chains without recursive data duplication.
* **Deterministic JSON Validation**: Uses `PydanticOutputParser` and `Literal` enums to guarantee the LLM outputs the exact schemas required by the backend.

## 🏗️ Pipeline Architecture

The pipeline analyzes raw text through 3 independent execution steps:

1. **Sentiment Analysis Chain**: Classifies the raw review into strict `Literal["positive", "negative", "neutral"]` categories.
2. **Summarization Chain**: Extracts 3-5 core bullet points representing the customer's primary concerns or praises.
3. **Response Generation Chain**: Synthesizes the contextual sentiment and summary into a highly personalized, empathetic customer support response.

## 🛠️ Technology Stack
* **Python**
* **LangChain Core**
* **OpenAI API (`gpt-4o-mini`)**
* **Pydantic**

## 💻 Example Output

```json
{
    "review": "Disappointed with this laptop. It's constantly overheating after just 30 minutes of use...",
    "sentiment": {
        "sentiment": "negative"
    },
    "summary": {
        "key_points": [
            "Laptop overheats after 30 minutes of use",
            "Battery life only lasts about 3 hours instead of the advertised 8"
        ]
    },
    "response": {
        "response": "Thank you for sharing your experience. I'm truly sorry to hear about the issues you've encountered with your laptop, including the overheating and battery life.
