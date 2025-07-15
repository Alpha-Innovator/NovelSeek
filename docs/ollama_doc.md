```

## Deploy LLMs using [ollama](https://ollama.com/) (Optional)

1. Download ollama following the [instructions](https://ollama.com/download) on the website.
2. (Optional) Run `export OLLAMA_MODELS=path/to/models` to specify the model storage path of ollama.
3. Run `ollama serve`.
4. Run `ollama run model_name`, for example, `ollama run deepseek-v2.5`. More model information can be found [here](https://ollama.com/library).

**Note that you need to modify the modelfile of ollama for more tokens**. For example,

1. Create a new file by `vim deepseek-v2.5-32k-modelfile`.
2. Modify the `num_ctx`, `num_predict` ... in the modelfile.
    ```
    FROM deepseek-v2.5
    PARAMETER num_ctx 24576
    PARAMETER num_predict 8192
    ```
3. Run `ollama create deepseek-v2.5-32k -f deepseek-v2.5-32k-modelfile`.
4. After that, you can run `ollama run deepseek-v2.5-32k`.

**Make aider support your own LLMs.**. For example,

1. You can use OpenAI API if you use ollama. For example,

    ```python
    import openai
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="na")
    ```

2. You need to add model setting to ensure the Coder class support your own LLMs. For example,

    In `/Path/to/env/lib/python3.x/site-packages/aider/models.py`
    
    ```python
    ModelSettings(
        "ollama/deepseek-v2.5-32k",
        "diff",
        use_repo_map=True,
        send_undo_reply=True,
        examples_as_sys_msg=True,
        reminder_as_sys_msg=True,
    )
    ```




