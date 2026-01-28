# Some Technical Details 

* llama.cpp is an open-source software library that allows users to run LLMs on local machines (including CPUs and GPUs) efficiently. 
    * Ollama is built on llama.cpp. 
* When temperature is set to zero, setting the seed for the language model does not matter. If we set the temperature to be larger than 0.0, then different seeds will yield different results.
  * Even with temperature at zero, the results might still vary slightly in different runs due to computational issues. When computers add numbers, the ordering might be slightly different depending on memory availability, batching, etc. in different runs. However, $(A + B) + C$ is not always equal to $A + (B + C)$ in floating-point math. As a result, we could still see slight variations in the language model's output.