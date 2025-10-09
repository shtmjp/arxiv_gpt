# Arxiv_gpt

- Summarize papers and send them to discord.
  - Use Google Gemini API
  - Avoid duplicates by recording posts to Spreadsheet
- Multiple arXiv queries can be configured with `EXTRA_ARXIV_SEARCH_CONFIGS`
- Replies with 「解説して」 now trigger an additional Gemini 2.5 Pro analysis that is
  converted into a PDF and uploaded to the same Discord channel.

## Explanation requests

- Set `DISCORD_BOT_TOKEN` so the bot can read replies and post files to the
  corresponding channel.
- Optionally override `EXPLANATION_STATE_PATH` to change where the bot stores
  metadata about posted summaries.

## Configuring multiple queries

Set the `EXTRA_ARXIV_SEARCH_CONFIGS` environment variable to a JSON array of
objects. Each object must include the query string and either a webhook URL or
the name of an environment variable that stores a webhook URL.

```json
[
  {
    "query": "cat:stat.ML",
    "webhook_env": "DISCORD_WEBHOOK_URL_STAT"
  },
  {
    "query": "ti:\"lead lag\"",
    "webhook_url": "https://discord.example/webhook"
  }
]
```

An example file is available at
[`extra_arxiv_search_configs.example.json`](./extra_arxiv_search_configs.example.json).
