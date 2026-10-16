# V0: understand the whole request

The outcome is deliberately small: give RepoPilot a public repository URL and get
six useful fields back: name, purpose, features, technologies, run instructions,
and limitations. You can follow the whole implementation without an agent framework.

## 1. Start the application

The shortest route is the Docker command in the README. Copy `.env.example` to
`.env`, put your own OpenAI API key there, then run `docker compose up --build`.
Open http://localhost:8501. The key stays in the API process; the browser does not
send it with each request. `.gitignore` excludes `.env`, and `.dockerignore`
excludes it from the image. Do not paste your key into Python files.

If learning locally, run FastAPI with `uvicorn repopilot.api:app --reload` and
Streamlit with `streamlit run repopilot/ui.py` in two terminals after installing
the requirements in a virtual environment. A virtual environment gives this
project its own installed packages. Run commands from the repository root.

A useful first example is `https://github.com/pallets/flask`. The GitHub fetch was
verified against that public repository; generating its summary requires your key.

If it fails: an unreachable API means start Uvicorn/check port 8000; a key error
means edit `.env` and restart the API; a GitHub rejection may be a rate limit or
an incorrect/private URL. Only public repositories are supported. No API key is
needed for the unit tests or the explicitly fake evaluation demo.

## 2. Fetch evidence before asking the model

Without repository context, an LLM can answer from old training knowledge or guess.
V0 improves on a bare question by supplying current metadata and the README.
This is direct context injection; we do not yet embed documents or search a database.

In `repopilot/github.py`:

```python
metadata = client.get(f"/repos/{owner}/{name}")
readme_response = client.get(f"/repos/{owner}/{name}/readme")
```

The input is an owner and repository name. `httpx` sends HTTP GET requests to
GitHub's fixed API host. GitHub returns JSON: named fields containing metadata
or a file wrapped in base64. HTTP is the request/response protocol; JSON is the
text format used to carry the fields. Neither is an LLM-specific concept.

```python
readme = base64.b64decode(document["content"]).decode("utf-8", errors="replace")
```

Base64 decoding recovers the file bytes; UTF-8 decoding turns those bytes into
Python text. We keep the source URL alongside that text. URL parsing restricts
input to an HTTPS GitHub owner/repository URL rather than fetching arbitrary hosts.

The naive approach sends the entire README. We take one small step: keep only its
first 20,000 characters and expose when clipping happened. This bounds prompt
size approximately, but characters are not tokens. An important instruction at
the end of a large README can be lost. Keep this simple baseline for V1 comparison.

Checkpoint: the GitHub decoding test uses a tiny fake API response to show the
byte-to-text conversion. The saved Flask source in `evals/results` is a real fetch.
Explain back: why is base64 decoding different from interpreting JSON?

## 3. Call the LLM and build its context

The simplest model call sends text and receives text. RepoPilot adds an application
instruction and evidence so the model knows what task to perform and what to use.
See `repopilot/llm.py`, especially `messages` and `client.responses.create(...)`.

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Summarise this UNTRUSTED REPOSITORY DATA:\n" + json.dumps(repository)},
]
```

The system message describes our task and uncertainty rules. The user message
carries the repository data. We label this data untrusted: a README may contain
instructions that should be described as content, not obeyed. This prompt is a
useful boundary, not a mathematical guarantee of truthful or injection-proof output.
The model receives no tools, credentials in the prompt, or repository write capability.

An LLM is trained to predict tokens from preceding context. Inference runs that
trained model to generate a response; this request does not train it on the repository.
A token is a piece of text, not necessarily a word. The context window is the
model's capacity for request and response tokens. More context can cost more and
still omit the needed fact. A low `temperature` reduces sampling randomness;
`temperature=0` is not a guarantee that repeated calls will be identical.

We use `max_output_tokens=2000` and a 45-second SDK timeout, with SDK retries
turned off. The Python loop allows at most two calls for schema correction;
network/API failures surface directly. This keeps the mechanism visible. Timeouts
bound waiting on network operations; they are not an exact whole-request deadline.

Model output can sound confident and still be wrong: hallucination means an
unsupported or invented claim here. Asking for “not stated” and showing limitations
helps, but evaluation is still needed. Prompt version `v0-summary-1` lets us track
which instructions produced a saved result.

Checkpoint: `test_real_sdk_serialization` exercises the real SDK against fake HTTP,
so no key is needed. A live request additionally needs an account/key and credits.
Explain back: what evidence does the model receive, and what information is absent?

## 4. Turn output into a predictable structure

A naive model answer is a paragraph. A UI cannot reliably find its installation
instructions if the layout changes each time. Pydantic declares the shape we need.
In `repopilot/models.py`:

```python
class RepositorySummary(BaseModel):
    name: str
    purpose: str
    main_features: list[str]
```

`BaseModel` is Pydantic's validated-data base class. Type annotations describe
expected values; `list[str]` means a list of strings. The full class also contains
technologies, how-to-run text and limitations. The OpenAI request receives the
JSON Schema generated by `RepositorySummary.model_json_schema()` to constrain
its output. We then validate again locally:

```python
summary = RepositorySummary.model_validate_json(response.output_text)
```

Input: generated JSON text. Output: a Python `RepositorySummary`, or an explicit
validation error. One correction request is allowed; two invalid outputs fail.
That is a bounded retry, not an agent. Required fields and strict types protect
the API contract, but a fabricated installation command is still a valid string.
When README is missing, code explicitly sets unknown run instructions and a warning.

Checkpoint: tests demonstrate valid output, invalid JSON, successful correction,
and the two-call stopping point. Explain back: why does schema-valid not mean true?

## 5. Expose the flow through FastAPI

`repopilot/api.py` joins the two operations:

```python
result = summarise(fetch_repository(request.repository_url))
```

FastAPI maps an HTTP POST at `/api/v1/repositories/summary` to the Python function.
The decorator above the function registers that mapping. FastAPI first validates
the JSON request using `SummaryRequest`; `response_model=SummaryResponse` defines
the JSON returned to the caller. A plain synchronous function is enough for V0;
FastAPI runs these blocking handlers in worker threads.

The API owns the flow and environment configuration. The fetching code owns GitHub
communication, and the summary code owns model communication. These are ordinary
functions, separated so their inputs and outputs are easy to inspect. There is no
database or persistent session: every Analyse request fetches again.

Checkpoint: open http://localhost:8000/docs and send the example request. Invalid
URLs produce validation errors; upstream failures produce a visible error response.
Explain back: why can both a browser UI and curl use the same API?

## 6. Display the result with Streamlit

`repopilot/ui.py` builds a form, waits for Analyse, sends the URL to FastAPI, and
renders the returned fields. Streamlit executes the Python script on interaction;
a form groups edits into one submitted request. The UI does not fetch GitHub or
call the model directly, so that logic has one home in the API.

`httpx.post(..., json={"repository_url": url})` is an ordinary HTTP client call.
The API returns JSON; `response.json()` makes it a Python dictionary. The expander
shows that exact response so you can connect UI fields to the backend data.
Generated run instructions are displayed as text, never executed.

Docker Compose runs two processes in containers. The UI container reaches the
API using the Compose service name `api`; your host browser uses `localhost`.
The Docker image packages Python and installed libraries. Compose supplies commands,
ports and configuration. Only the API container receives `.env` secrets.
CI repeats install → tests → image build on GitHub; a local pass is not a remote CI result.

Checkpoint: open port 8501 and compare its fields with `/docs` or curl.
Explain back: why does `localhost` inside the UI container differ from your host?

## 7. Inspect a small baseline before adding complexity

Run `python -m pytest -q` for focused software checks. Mocks replace external
services with fixed responses; they test our flow without proving real model quality.
The Streamlit check simulates a form submission. There is no coverage target or
performance optimization exercise in V0.

`evals/repository.json` is a deliberately synthetic notes-app README, not a real
GitHub project. `evals/questions.json` contains ten expected-fact questions about
one summary. `python -m evals.run --output evals/results/my-live-baseline.json`
calls the model once and saves the source, output, model/prompt settings, usage,
time and review slots. It never overwrites an existing result file. Inspect each
question and mark whether the output is correct and contains unsupported claims.
Also check the full answer for invented details beyond those ten checks.

The `--demo` option uses handwritten fake output to teach the file format. Its
saved tokens/time are not real LLM performance. It has no answer-quality score.
API token counts come from the provider; cost is left unmeasured rather than
assuming model prices. Later, estimate cost from actual usage and dated prices.

This baseline asks: does a direct summary reflect the supplied README? It cannot
establish general answer accuracy from one small synthetic source. Next, V1 will
introduce chunking, embeddings, vector search and retrieval evaluation to address
context-selection limits. Preserve the V0 prompt and outputs for comparison.

Interview explanation to practise: “The UI sends a URL to FastAPI. Python fetches
GitHub JSON, decodes the README, and places it beside task instructions in the
model context. The API requests a JSON schema and validates the result with
Pydantic before returning it. That guarantees shape, not factual truth, so I
inspect generated facts against fixed evidence and record the baseline.”

Official references used for the API boundaries: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses) and [GitHub repository contents/README](https://docs.github.com/en/rest/repos/contents#get-a-repository-readme).

## Troubleshooting: OpenAI credits exhausted

If Analyse reports “OpenAI API credits/quota are exhausted”, Docker and the key
loading may be working correctly. OpenAI can return HTTP 429 with code
`credit_balance_exhausted` and type `insufficient_quota`. This means API access
needs available credits; repeatedly retrying the same request will not fix it.
Check the API billing account associated with your key. See the
[official error guide](https://developers.openai.com/api/docs/guides/error-codes).
A temporary request-rate limit also uses 429, but asks you to wait instead.

The API now distinguishes these cases with safe messages. It does not return raw
upstream exception text, which can include sensitive request details. If you change
`.env`, run `docker compose up -d --force-recreate api` so the container receives
its new environment. A plain process restart does not re-read Compose's env file.
