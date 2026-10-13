import requests
import json

def test_query(query: str):
    url = "http://127.0.0.1:8543/api/chat"
    print(f"Testing query: {query}")
    try:
        response = requests.post(url, json={"query": query}, timeout=60, stream=True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[6:]
                        data = json.loads(data_str)
                        if "error" in data:
                            print(f"Error: {data['error']}")
                        elif "chunk" in data and not "done" in data:
                            pass # Streaming chunk, ignore for summary
                        elif "done" in data:
                            print("\n--- Answer ---")
                            print(data.get("final_verified_response", ""))
                            
                            print("\n--- Retrieved Context ---")
                            context = data.get("citations", [])
                            for i, c in enumerate(context):
                                payload = c.get("payload", {})
                                print(f"[{i+1}] {payload.get('filename')} (Score: {c.get('score')}):")
                                print(payload.get("content", "")[:200].replace('\n', ' ') + "...")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    queries = [
        "What is the eligible limit for inclusion of Perpetual Debt Instruments (PDIs) issued in foreign currency in Additional Tier 1 (AT1) capital?",
        "Which specific circular regarding Perpetual Debt Instruments (PDI) in Additional Tier 1 Capital was repealed on September 29, 2025?",
        "Explain the tier 1 capital requirements under Basel III and how PDIs are factored in."
    ]
    for q in queries:
        test_query(q)
        print("\n" + "="*80 + "\n")
