import asyncio
import aiohttp
import time
from tqdm import tqdm

URL = "https://aiskripsisbisindo-production.up.railway.app/predict_landmarks"

# Contoh data JSON dummy
payload = {"landmarks": [0.1] * 126}
headers = {"Content-Type": "application/json"}

# Jumlah user dan request per user
TOTAL_USERS = 50
REQUESTS_PER_USER = 5

async def send_request(session, user_id):
    latencies = []
    for _ in range(REQUESTS_PER_USER):
        start = time.time()
        async with session.post(URL, json=payload, headers=headers) as response:
            await response.text()
        latencies.append(time.time() - start)
    return latencies

async def run_load_test():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(TOTAL_USERS)]
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.extend(await f)

    avg_time = sum(results) / len(results)
    max_time = max(results)
    print(f"\n✅ Total Requests: {len(results)}")
    print(f"⚡ Average Response Time: {avg_time:.3f} s")
    print(f"🔥 Max Response Time: {max_time:.3f} s")

if __name__ == "__main__":
    asyncio.run(run_load_test())
