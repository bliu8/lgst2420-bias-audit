# AI Bias Audit Dashboard
### LGST 2420 Final Project — Option 2: Build

A tool that tests AI models for demographic bias. It sends identical prompts to an AI model, changing only the name of the person being evaluated, and measures whether the model's responses differ by race or gender.

---

## What This Tool Does

The dashboard runs a **correspondence audit** — the same method used in academic research to study discrimination in hiring. A scenario (hiring decision, loan application, or medical triage) is sent to an AI model dozens of times. The only thing that changes between requests is the candidate's name. Names were chosen from audit literature because they reliably signal race and gender to readers.

If the AI responds differently depending on the name — recommending one group more often, using different language, or refusing to answer more for some groups — the tool flags it and shows you exactly where and by how much.

---

## Setup (Step by Step)

### Step 1 — Make sure Python is installed

Open your Terminal (Mac) or Command Prompt (Windows) and type:

```
python --version
```

You should see something like `Python 3.10.0` or higher. If you get an error, download Python from [python.org](https://python.org) and install it first.

---

### Step 2 — Download the project

If you have git installed:
```
git clone <your-repo-url>
cd LGST_2420_final_project
```

Or download the ZIP from GitHub, unzip it, and open a terminal in that folder.

---

### Step 3 — Create a virtual environment

This keeps the project's dependencies separate from the rest of your computer. Run these commands one at a time:

**Mac/Linux:**
```
python -m venv .venv
source .venv/bin/activate
```

**Windows:**
```
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` appear at the start of your terminal line. That means it worked.

---

### Step 4 — Install dependencies

```
pip install -r requirements.txt
```

This installs all the libraries the project needs. It may take a minute.

---

### Step 5 — Set up your API key

The tool needs an Anthropic API key to call the AI model. Here is how to get one and set it up:

1. Go to [console.anthropic.com](https://console.anthropic.com) and create a free account
2. Click **API Keys** in the left sidebar
3. Click **Create Key**, give it a name, and copy the key (it starts with `sk-ant-...`)

Now set up the key file:

```
cp .env.example .env
```

This creates a file called `.env` in the project folder. Open it with any text editor (TextEdit on Mac, Notepad on Windows). It will look like this:

```
ANTHROPIC_API_KEY=your_api_key_here
AUDIT_MODEL=claude-haiku-4-5-20251001
RESULTS_DIR=data/results
```

Replace `your_api_key_here` with the key you copied. Leave the other two lines as they are. Save the file.

> **Important:** Never share your `.env` file or post it online. It contains your private API key. The project's `.gitignore` file already prevents it from being uploaded to GitHub automatically.

---

### Step 6 — Launch the dashboard

```
streamlit run dashboard/app.py
```

Your browser should open automatically to `http://localhost:8501`. If it does not, open your browser and go to that address manually.

---

## Using the Dashboard

### Running an audit

1. Click the **Run Audit** tab
2. Your API key will be pre-filled if you set up the `.env` file. Otherwise paste it in.
3. Choose a **Scenario** (Hiring, Lending, or Medical)
4. Choose a **Model** — Haiku 4.5 is recommended, it is the fastest and cheapest
5. Set the **Number of probes** — how many prompts to send. More probes means more reliable results. Use at least 40 for any meaningful conclusion.
6. Click **Run audit** and wait for the progress bar to complete
7. When done, the results load automatically and a report download button appears

### Reading the results

- **Overview tab** — headline metrics: total probes, errors, aggregate disparity score, and a status indicator
- **By Group tab** — four charts showing positive rates, deviations from average, disparity gaps, and refusal rates per demographic group. An automatic interpretation box tells you whether a disparity was detected.
- **Completions tab** — the actual text of every model response. Filter by subgroup, outcome type, or flagged groups only to compare responses side by side.

### Downloading a report

After an audit completes, a **Download Markdown report** button appears in the Run Audit tab. The report contains a summary of findings, a metrics table, and flagged completions in a format suitable for a governance memo.

---

## Running Tests

To verify the tool is working correctly (no API key required):

```
pytest tests/ -v -m "not integration"
```

---

## Project Structure

```
audit/          Core audit library (probes, metrics, runner, report)
dashboard/      Streamlit UI
data/prompts/   Probe prompt templates
data/results/   Audit output JSON files (not uploaded to GitHub)
tests/          Unit and integration tests
```

