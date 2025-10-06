SUMMARY_PREFIX = """
与えられた論文の要点を3点のみでまとめ、以下のフォーマットで日本語で出力してください。\n
```
・要点1
・要点2
・要点3
```

注意事項:
* 厳格に上記フォーマットに従うこと(「承知しました」などは不要)
* 与えられた文書注の専門用語は, 日本語(原文ママの英語)とすること
"""

SEARCH_QUERY_LLAG = "ti: %22 lead lag %22 OR abs: %22 lead lag %22"
# "lead-lag" does not work maybe due to the hyphen
SEARCH_QUERY = "cat: math.ST"
