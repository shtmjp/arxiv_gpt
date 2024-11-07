SUMMARY_PREFIX = """
与えられた論文の要点を3点のみでまとめ、以下のフォーマットで日本語で出力してください。\n
```
・要点1
・要点2
・要点3
```
"""

SEARCH_QUERY_LLAG = "ti: %22 lead lag %22 OR abs: %22 lead lag %22"
# "lead-lag" does not work maybe due to the hyphen
SEARCH_QUERY = "cat: math.ST"
