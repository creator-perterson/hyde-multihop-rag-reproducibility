# GitHub Upload Steps

Run these commands from the repository root. Use an anonymous local Git identity for all review-release commits:

```powershell
cd <repo-root>
git init
git config user.name "Anonymous Authors"
git config user.email "anonymous@example.com"
git status
git add .
git commit -m "Initial reproducibility release"
```

Create a new empty repository on GitHub, then connect and push:

```powershell
git branch -M main
git remote add origin https://github.com/creator-perterson/hyde-multihop-rag-reproducibility.git
git push -u origin main
```

If a non-anonymous initial commit was already pushed, rewrite the local initial commit with the anonymous identity and push with `--force-with-lease` after confirming this is allowed by the review policy.

Before making the repository public:

```powershell
git status
rg -n "api_key|apikey|secret|password|Authorization|BEGIN .*PRIVATE|LLM_API_KEY=|OPENAI_API_KEY=" .
```

Expected secret-scan hits should be limited to placeholders in `.env.example`, documentation, or code that reads environment variables. Do not publish real `.env` files, raw credentials, provider request exports, or private screenshots.

For double-blind review, keep the repository private or anonymized until the venue policy allows public release.
