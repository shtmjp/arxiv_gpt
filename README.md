# Arxiv_gpt

- Summarize papers and send them to discord.
  - Use Google Gemini API
  - Avoid duplicates by recording posts to Spreadsheet
- Multiple arXiv queries can be configured with `EXTRA_ARXIV_SEARCH_CONFIGS`

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
