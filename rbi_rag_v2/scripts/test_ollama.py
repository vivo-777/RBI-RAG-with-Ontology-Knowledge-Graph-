import dspy

try:
    lm = dspy.LM("ollama_chat/qwen2.5:7b-instruct", api_base="http://localhost:11434")
    with dspy.context(lm=lm):
        print(lm("Hello! What is 2+2?"))
except Exception as e:
    print("Error:", e)
