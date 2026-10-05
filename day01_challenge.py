import random
import asyncio
import time

async def process_prompt(prompt_id:int, text:str) -> dict:
    print(f"[START] processing prompt {prompt_id}...")
    await asyncio.sleep(random.uniform(1,3))
    return {
        "id": prompt_id,
        "text": text,
        "status": "completed"
    }

async def main():
    prompts = [
        (1, "Summarize article"),
        (2, "Translate to Hindi"),
        (3, "Extract keywords"),
        (4, "generate a image"),
        (5, "generate video")
    ]

    start_time = time.perf_counter()
    
    tasks = [process_prompt(id, text) for id, text in prompts]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.perf_counter() - start_time

    print("------Printing Results------")    
    for res in results:
        print(res)

    print(f"Total time elapsed: {elapsed}")    

if __name__ == "__main__":
    asyncio.run(main())    


