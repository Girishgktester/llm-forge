from langchain_core.runnables import RunnableLambda, RunnableParallel

# 1. RunnableLambda → clean requirement
clean = RunnableLambda(lambda x: x.strip())

# 2. RunnableLambda → generate simple outputs
functional = RunnableLambda(
    lambda x: f"Functional test case for: {x}"
)

negative = RunnableLambda(
    lambda x: f"Negative test case for: {x}"
)

# 3. RunnableParallel → run both independently
parallel = RunnableParallel(
    functional=functional,
    negative=negative
)

# 4. Complete Runnable flow
chain = clean | parallel

# 5. Invoke
result = chain.invoke(
    "  User should be able to login with valid credentials  "
)

print(result)