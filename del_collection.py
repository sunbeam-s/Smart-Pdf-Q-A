import inspect
from inngest.experimental import ai
print(inspect.signature(ai.openai.Adapter.__init__))