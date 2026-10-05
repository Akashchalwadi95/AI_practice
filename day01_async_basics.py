import asyncio
import time

# Simulated asynchronous call to an LLM provider (API network latency)
async def fetch_llm_response(provider: str, delay: int) -> dict:
    print(f"[{time.strftime('%X')}] Sending request to {provider}...")
    await asyncio.sleep(delay)  # Non-blocking pause that yields control back to event loop
    print(f"[{time.strftime('%X')}] Received response from {provider}!")
    return {"provider": provider, "status": "success", "latency": delay}

async def main():
    start_time = time.perf_counter()

    providers = [
        ("Gemini", 2),
        ("OpenAI", 3),
        ("Anthropic", 1),
    ]

    print("--- Starting Concurrent LLM Calls ---")
    
    # Launch all requests concurrently using asyncio.gather
    tasks = [fetch_llm_response(name, delay) for name, delay in providers]
    results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    
    print("\n--- Results ---")
    for res in results:
        print(res)
        
    print(f"\nTotal Execution Time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())